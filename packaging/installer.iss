; Tennis Tagger — Inno Setup script
; Compile with Inno Setup 6: ISCC.exe packaging\installer.iss
; Produces: dist\TennisTagger-Setup-<version>.exe

#define MyAppName       "Tennis Tagger"
#define MyAppVersion    "0.2.2"
#define MyAppPublisher  "Liam Parker"
#define MyAppExeName    "TennisTagger.bat"
#define MyBundleDir     "..\dist\TennisTagger"

[Setup]
AppId={{8C2F9E14-7B6D-4A6B-9F1E-7C0A1A0B2D33}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
; Per-user install only. The app writes to its own dir at runtime
; (sync_config.json, model backups, runs/), which fails under Program Files
; without elevation. Forcing %LOCALAPPDATA% means no UAC prompt and no
; PermissionError surprises.
DefaultDirName={localappdata}\Programs\TennisTagger
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
ArchitecturesInstallIn64BitMode=x64compatible
ArchitecturesAllowed=x64compatible
PrivilegesRequired=lowest
OutputDir=..\dist
OutputBaseFilename=TennisTagger-Setup-{#MyAppVersion}
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
DisableDirPage=no
DisableReadyPage=no
ShowLanguageDialog=no
LicenseFile=
SetupIconFile=
UninstallDisplayName={#MyAppName} {#MyAppVersion}

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"
Name: "quicklaunchicon"; Description: "{cm:CreateQuickLaunchIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked; OnlyBelowVersion: 6.1

[Files]
; Recursively pull in everything the build script staged.
Source: "{#MyBundleDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}";        Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"
Name: "{group}\Uninstall {#MyAppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}";   Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#MyAppName}}"; Flags: nowait postinstall skipifsilent shellexec

[UninstallDelete]
; Remove leftover caches & runs the user generated post-install
Type: filesandordirs; Name: "{app}\python\Lib\site-packages\__pycache__"
Type: filesandordirs; Name: "{app}\runs"
Type: filesandordirs; Name: "{app}\__pycache__"
Type: files;          Name: "{app}\launcher.log"
Type: files;          Name: "{app}\sync_config.json"

[Code]
function InitializeSetup(): Boolean;
var
  Version: TWindowsVersion;
begin
  GetWindowsVersionEx(Version);
  if Version.Major < 10 then
  begin
    MsgBox('Tennis Tagger requires Windows 10 or later.', mbError, MB_OK);
    Result := False;
  end
  else
    Result := True;
end;
