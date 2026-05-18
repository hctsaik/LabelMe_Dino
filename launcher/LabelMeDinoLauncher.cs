using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;

namespace LabelMeDinoLauncher
{
    internal static class Program
    {
        private static int Main(string[] args)
        {
            try
            {
                string exeDir = AppDomain.CurrentDomain.BaseDirectory.TrimEnd(Path.DirectorySeparatorChar);
                string appRoot = FindAppRoot(exeDir);
                string runtime = FindRuntime(exeDir, appRoot);
                string python = Path.Combine(runtime, "Scripts", "python.exe");

                if (!File.Exists(python))
                {
                    Console.Error.WriteLine("python.exe not found: " + python);
                    return 2;
                }

                if (args.Contains("--probe-runtime"))
                {
                    return RunProbe(python, runtime, appRoot);
                }

                string mainPy = Path.Combine(appRoot, "main.py");
                if (!File.Exists(mainPy))
                {
                    Console.Error.WriteLine("main.py not found: " + mainPy);
                    return 3;
                }

                var forwardedArgs = new List<string> { Quote(mainPy) };
                forwardedArgs.AddRange(args.Select(Quote));

                return RunPython(python, string.Join(" ", forwardedArgs), runtime, appRoot, inheritStdIo: false);
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine(ex.ToString());
                return 1;
            }
        }

        private static string FindAppRoot(string exeDir)
        {
            var candidates = new[]
            {
                exeDir,
                Path.Combine(exeDir, "app"),
                Directory.GetParent(exeDir) != null ? Directory.GetParent(exeDir).FullName : exeDir
            };
            foreach (string candidate in candidates)
            {
                if (File.Exists(Path.Combine(candidate, "main.py")) &&
                    Directory.Exists(Path.Combine(candidate, "src")))
                {
                    return candidate;
                }
            }
            return exeDir;
        }

        private static string FindRuntime(string exeDir, string appRoot)
        {
            var candidates = new List<string>();
            string envRuntime = Environment.GetEnvironmentVariable("LABELME_DINO_RUNTIME");
            if (!string.IsNullOrWhiteSpace(envRuntime))
            {
                candidates.Add(envRuntime);
            }

            string localAppData = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
            if (!string.IsNullOrWhiteSpace(localAppData))
            {
                candidates.Add(Path.Combine(localAppData, "CIM", "labelme-dino-runtime", ".venv"));
            }

            candidates.Add(Path.Combine(exeDir, ".venv"));
            candidates.Add(Path.Combine(exeDir, "runtime", ".venv"));
            candidates.Add(Path.Combine(appRoot, ".venv"));
            if (Directory.GetParent(exeDir) != null)
            {
                candidates.Add(Path.Combine(Directory.GetParent(exeDir).FullName, ".venv"));
            }
            var cursor = new DirectoryInfo(exeDir);
            while (cursor != null)
            {
                candidates.Add(Path.Combine(cursor.FullName, "LabelMe_Dino", ".venv"));
                candidates.Add(Path.Combine(cursor.FullName, ".venv"));
                cursor = cursor.Parent;
            }

            foreach (string candidate in candidates)
            {
                if (File.Exists(Path.Combine(candidate, "Scripts", "python.exe")) &&
                    Directory.Exists(Path.Combine(candidate, "Lib", "site-packages")))
                {
                    return Path.GetFullPath(candidate);
                }
            }

            throw new DirectoryNotFoundException(
                "LabelMe_Dino runtime not found. Set LABELME_DINO_RUNTIME to a .venv folder.");
        }

        private static int RunProbe(string python, string runtime, string appRoot)
        {
            string code =
                "import json, sys; " +
                "import torch, cv2, transformers; " +
                "from PyQt5.QtCore import QT_VERSION_STR; " +
                "print(json.dumps({" +
                "'ok': True, " +
                "'python': sys.version.split()[0], " +
                "'torch': getattr(torch, '__version__', 'unknown'), " +
                "'cuda_available': bool(torch.cuda.is_available()), " +
                "'cv2': getattr(cv2, '__version__', 'unknown'), " +
                "'transformers': getattr(transformers, '__version__', 'unknown'), " +
                "'qt': QT_VERSION_STR" +
                "}))";
            return RunPython(python, "-c " + Quote(code), runtime, appRoot, inheritStdIo: true);
        }

        private static int RunPython(
            string python,
            string arguments,
            string runtime,
            string appRoot,
            bool inheritStdIo)
        {
            var psi = new ProcessStartInfo
            {
                FileName = python,
                Arguments = arguments,
                WorkingDirectory = appRoot,
                UseShellExecute = false,
                RedirectStandardOutput = !inheritStdIo,
                RedirectStandardError = !inheritStdIo,
                CreateNoWindow = false
            };

            PrepareEnvironment(psi, runtime);

            using (var proc = Process.Start(psi))
            {
                if (proc == null)
                {
                    Console.Error.WriteLine("Failed to start python.");
                    return 4;
                }

                if (!inheritStdIo)
                {
                    proc.OutputDataReceived += (sender, e) => { if (e.Data != null) Console.Out.WriteLine(e.Data); };
                    proc.ErrorDataReceived += (sender, e) => { if (e.Data != null) Console.Error.WriteLine(e.Data); };
                    proc.BeginOutputReadLine();
                    proc.BeginErrorReadLine();
                }

                proc.WaitForExit();
                return proc.ExitCode;
            }
        }

        private static void PrepareEnvironment(ProcessStartInfo psi, string runtime)
        {
            string sitePackages = Path.Combine(runtime, "Lib", "site-packages");
            string torchLib = Path.Combine(sitePackages, "torch", "lib");
            string qtBin = Path.Combine(sitePackages, "PyQt5", "Qt5", "bin");
            string qtPlugins = Path.Combine(sitePackages, "PyQt5", "Qt5", "plugins");
            string cv2Dir = Path.Combine(sitePackages, "cv2");
            string scripts = Path.Combine(runtime, "Scripts");

            string oldPath = psi.EnvironmentVariables.ContainsKey("PATH")
                ? psi.EnvironmentVariables["PATH"]
                : Environment.GetEnvironmentVariable("PATH");
            psi.EnvironmentVariables["PATH"] = string.Join(";", new[]
            {
                scripts,
                torchLib,
                qtBin,
                cv2Dir,
                oldPath
            }.Where(p => !string.IsNullOrWhiteSpace(p) && Directory.Exists(p)));

            psi.EnvironmentVariables["PYTHONUTF8"] = "1";
            psi.EnvironmentVariables["LABELME_DINO_RUNTIME"] = runtime;
            psi.EnvironmentVariables["PYTHONPATH"] = sitePackages;
            if (Directory.Exists(qtPlugins))
            {
                psi.EnvironmentVariables["QT_PLUGIN_PATH"] = qtPlugins;
                psi.EnvironmentVariables["QT_QPA_PLATFORM_PLUGIN_PATH"] = Path.Combine(qtPlugins, "platforms");
            }
        }

        private static string Quote(string value)
        {
            if (value == null)
            {
                return "\"\"";
            }
            return "\"" + value.Replace("\\", "\\\\").Replace("\"", "\\\"") + "\"";
        }
    }
}
