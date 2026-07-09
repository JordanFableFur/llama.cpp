# llama.cpp fork launcher -- UX PROTOTYPE (PowerShell + WPF, zero-install)
# Double-click run-prototype.cmd to launch. Throwaway mockup to settle the layout before the
# real signed WPF build. Spec: publications/LAUNCHER-SPEC.md. "A launcher, not a studio."
# ASCII-only UI strings by rule -- no non-ASCII glyphs anywhere in this file.

Add-Type -AssemblyName PresentationFramework
Add-Type -AssemblyName System.Windows.Forms

$defaultServer  = "C:\Users\Jorda\llama.cpp\build-main\bin\Release\llama-server.exe"
$defaultModels  = "C:\Users\Jorda\models"

[xml]$xaml = @"
<Window xmlns="http://schemas.microsoft.com/winfx/2006/xaml/presentation"
        xmlns:x="http://schemas.microsoft.com/winfx/2006/xaml"
        Title="llama.cpp launcher (fork) - prototype" Height="700" Width="620"
        WindowStartupLocation="CenterScreen" Background="#FF1E1E1E">
  <Window.Resources>
    <Style TargetType="TextBlock"><Setter Property="Foreground" Value="#FFE0E0E0"/>
      <Setter Property="VerticalAlignment" Value="Center"/><Setter Property="Margin" Value="0,0,8,0"/></Style>
    <Style TargetType="CheckBox"><Setter Property="Foreground" Value="#FFE0E0E0"/>
      <Setter Property="VerticalAlignment" Value="Center"/></Style>
    <Style TargetType="GroupBox"><Setter Property="Foreground" Value="#FF7FB2E5"/>
      <Setter Property="Margin" Value="0,0,0,10"/><Setter Property="Padding" Value="8"/>
      <Setter Property="BorderBrush" Value="#FF3A3A3A"/></Style>
    <Style TargetType="Button"><Setter Property="Padding" Value="8,2"/></Style>
  </Window.Resources>

  <Grid Margin="14">
    <Grid.RowDefinitions>
      <RowDefinition Height="Auto"/><RowDefinition Height="Auto"/><RowDefinition Height="*"/><RowDefinition Height="Auto"/>
    </Grid.RowDefinitions>

    <StackPanel Grid.Row="0">
      <!-- MODEL -->
      <GroupBox Header="Model">
        <Grid>
          <Grid.RowDefinitions><RowDefinition Height="Auto"/><RowDefinition Height="Auto"/></Grid.RowDefinitions>
          <Grid.ColumnDefinitions>
            <ColumnDefinition Width="90"/><ColumnDefinition Width="*"/>
            <ColumnDefinition Width="Auto"/><ColumnDefinition Width="Auto"/>
          </Grid.ColumnDefinitions>
          <TextBlock Grid.Row="0" Grid.Column="0" Text="Models folder"/>
          <TextBox   Grid.Row="0" Grid.Column="1" Name="Folder" Margin="0,4"/>
          <Button    Grid.Row="0" Grid.Column="2" Name="BrowseFolder" Content="Browse" Margin="6,4,0,4"/>
          <Button    Grid.Row="0" Grid.Column="3" Name="Scan" Content="Scan" Margin="6,4,0,4"/>
          <TextBlock Grid.Row="1" Grid.Column="0" Text="Model"/>
          <ComboBox  Grid.Row="1" Grid.Column="1" Grid.ColumnSpan="3" Name="Model" Margin="0,4"/>
        </Grid>
      </GroupBox>

      <!-- CONFIG -->
      <GroupBox Header="Config">
        <StackPanel>
          <Grid Margin="0,0,0,6">
            <Grid.ColumnDefinitions><ColumnDefinition Width="90"/><ColumnDefinition Width="*"/></Grid.ColumnDefinitions>
            <TextBlock Grid.Column="0" Text="Profile"/>
            <ComboBox  Grid.Column="1" Name="Preset">
              <ComboBoxItem Content="Bigger-than-VRAM MoE (offload)"/>
              <ComboBoxItem Content="Fits-in-VRAM (dense / small)"/>
              <ComboBoxItem Content="Custom (no overrides)" IsSelected="True"/>
            </ComboBox>
          </Grid>
          <Grid Margin="0,2">
            <Grid.ColumnDefinitions>
              <ColumnDefinition Width="Auto"/><ColumnDefinition Width="60"/>
              <ColumnDefinition Width="Auto"/><ColumnDefinition Width="60"/>
              <ColumnDefinition Width="Auto"/><ColumnDefinition Width="70"/>
              <ColumnDefinition Width="Auto"/><ColumnDefinition Width="70"/>
              <ColumnDefinition Width="*"/>
            </Grid.ColumnDefinitions>
            <TextBlock Grid.Column="0" Text="Port"/>  <TextBox Grid.Column="1" Name="Port" Text="8080"/>
            <TextBlock Grid.Column="2" Text="-ngl" Margin="14,0,8,0"/><TextBox Grid.Column="3" Name="Ngl" Text="99"/>
            <TextBlock Grid.Column="4" Text="-ncmoe" Margin="14,0,8,0"/><TextBox Grid.Column="5" Name="Ncmoe" Text=""/>
            <TextBlock Grid.Column="6" Text="-ub" Margin="14,0,8,0"/><TextBox Grid.Column="7" Name="Ub" Text=""/>
          </Grid>
          <Grid Margin="0,4">
            <Grid.ColumnDefinitions>
              <ColumnDefinition Width="Auto"/><ColumnDefinition Width="70"/>
              <ColumnDefinition Width="Auto"/><ColumnDefinition Width="70"/><ColumnDefinition Width="*"/>
            </Grid.ColumnDefinitions>
            <TextBlock Grid.Column="0" Text="Ctx -c"/><TextBox Grid.Column="1" Name="Ctx" Text="4096"/>
            <TextBlock Grid.Column="2" Text="Threads -t" Margin="14,0,8,0"/><TextBox Grid.Column="3" Name="Threads" Text=""/>
          </Grid>
          <StackPanel Orientation="Horizontal" Margin="0,8,0,4">
            <CheckBox Name="Fa" Content="flash-attn (-fa)" IsChecked="True" Margin="0,0,24,0"/>
            <CheckBox Name="Pin" Content="host pinning (GGML_CUDA_REGISTER_HOST=1), ~2.2x prefill" IsChecked="False"/>
          </StackPanel>
          <Grid Margin="0,2">
            <Grid.ColumnDefinitions><ColumnDefinition Width="90"/><ColumnDefinition Width="*"/></Grid.ColumnDefinitions>
            <TextBlock Grid.Column="0" Text="Extra flags"/>
            <TextBox   Grid.Column="1" Name="Extra" ToolTip="Appended verbatim to the command line"/>
          </Grid>
        </StackPanel>
      </GroupBox>

      <!-- SERVER -->
      <GroupBox Header="Server binary">
        <Grid>
          <Grid.ColumnDefinitions><ColumnDefinition Width="90"/><ColumnDefinition Width="*"/><ColumnDefinition Width="Auto"/></Grid.ColumnDefinitions>
          <TextBlock Grid.Column="0" Text="llama-server"/>
          <TextBox   Grid.Column="1" Name="Server"/>
          <Button    Grid.Column="2" Name="BrowseServer" Content="Browse" Margin="6,0,0,0"/>
        </Grid>
      </GroupBox>
    </StackPanel>

    <Border Grid.Row="1" Name="PreviewBox" Visibility="Collapsed" Background="#FF2A2A2A" Margin="0,0,0,6" Padding="8">
      <TextBlock Name="Cmd" FontFamily="Consolas" TextWrapping="Wrap" Foreground="#FF9CDCFE" Text="(command preview)"/>
    </Border>

    <TextBox Grid.Row="2" Name="Log" Background="#FF141414" Foreground="#FFB5CEA8"
             FontFamily="Consolas" FontSize="11" IsReadOnly="True"
             VerticalScrollBarVisibility="Auto" TextWrapping="Wrap"
             Text="Ready. Set a models folder and hit Scan to list models."/>

    <Grid Grid.Row="3" Margin="0,8,0,0">
      <Grid.ColumnDefinitions><ColumnDefinition Width="*"/><ColumnDefinition Width="Auto"/></Grid.ColumnDefinitions>
      <CheckBox Grid.Column="0" Name="AdvancedToggle" Content="Advanced (show command)" VerticalAlignment="Center"/>
      <StackPanel Grid.Column="1" Orientation="Horizontal" HorizontalAlignment="Right">
        <Button Name="Launch" Content="Launch" Width="110" Height="32" Margin="0,0,8,0"/>
        <Button Name="Stop" Content="Stop" Width="80" Height="32" IsEnabled="False"/>
      </StackPanel>
    </Grid>
  </Grid>
</Window>
"@

$reader = New-Object System.Xml.XmlNodeReader $xaml
$win = [Windows.Markup.XamlReader]::Load($reader)

$ctrls = @{}
foreach ($n in 'Folder','BrowseFolder','Scan','Model','Preset','Port','Ngl','Ncmoe','Ub','Ctx','Threads','Fa','Pin','Extra','Server','BrowseServer','Cmd','PreviewBox','Log','AdvancedToggle','Launch','Stop') {
  $ctrls[$n] = $win.FindName($n)
}
$ctrls.Server.Text = $defaultServer
if (Test-Path $defaultModels) { $ctrls.Folder.Text = $defaultModels }

$script:proc = $null; $script:outFile = $null; $script:errFile = $null; $script:opened = $false; $script:timer = $null

function Log($msg) { $ctrls.Log.AppendText("`n$msg"); $ctrls.Log.ScrollToEnd() }

function Get-ModelPath { if ($ctrls.Model.SelectedItem) { return $ctrls.Model.SelectedItem.Path } return $null }

function Scan-Models {
  $folder = $ctrls.Folder.Text
  if (-not $folder -or -not (Test-Path $folder)) { Log "! models folder not found: $folder"; return }
  $files = Get-ChildItem -Path $folder -Recurse -Filter *.gguf -File -ErrorAction SilentlyContinue |
    Where-Object { -not ($_.Name -match '-(\d{5})-of-\d{5}\.gguf$' -and $Matches[1] -ne '00001') }
  $items = @($files | ForEach-Object {
    $disp = $_.FullName.Substring($folder.TrimEnd('\').Length).TrimStart('\')
    [pscustomobject]@{ Display = $disp; Path = $_.FullName }
  } | Sort-Object Display)
  $ctrls.Model.ItemsSource = $items
  $ctrls.Model.DisplayMemberPath = 'Display'
  if ($items.Count -gt 0) { $ctrls.Model.SelectedIndex = 0 }
  Log "found $($items.Count) model(s) in $folder"
  Refresh-Preview
}

function Build-Args {
  $a = @()
  $m = Get-ModelPath
  if ($m) { $a += @('-m', "`"$m`"") }
  $a += @('--port', $ctrls.Port.Text, '-ngl', $ctrls.Ngl.Text)
  if ($ctrls.Ncmoe.Text)   { $a += @('-ncmoe', $ctrls.Ncmoe.Text) }
  if ($ctrls.Ub.Text)      { $a += @('-b', '4096', '-ub', $ctrls.Ub.Text) }
  if ($ctrls.Ctx.Text)     { $a += @('-c', $ctrls.Ctx.Text) }
  if ($ctrls.Threads.Text) { $a += @('-t', $ctrls.Threads.Text) }
  $a += @('-fa', $(if ($ctrls.Fa.IsChecked) { '1' } else { '0' }))
  if ($ctrls.Extra.Text)   { $a += $ctrls.Extra.Text }
  return $a
}

function Refresh-Preview {
  $prefix = if ($ctrls.Pin.IsChecked) { 'GGML_CUDA_REGISTER_HOST=1 ' } else { '' }
  $ctrls.Cmd.Text = $prefix + '"' + $ctrls.Server.Text + '" ' + ((Build-Args) -join ' ')
}

$ctrls.Preset.Add_SelectionChanged({
  switch ($ctrls.Preset.SelectedIndex) {
    0 { $ctrls.Ncmoe.Text='22'; $ctrls.Ub.Text='2048'; $ctrls.Fa.IsChecked=$true; $ctrls.Pin.IsChecked=$true }
    1 { $ctrls.Ncmoe.Text='';   $ctrls.Ub.Text='';     $ctrls.Fa.IsChecked=$true; $ctrls.Pin.IsChecked=$false }
    default { }
  }
  Refresh-Preview
})
foreach ($f in 'Folder','Server','Port','Ngl','Ncmoe','Ub','Ctx','Threads','Extra') { $ctrls[$f].Add_TextChanged({ Refresh-Preview }) }
$ctrls.Model.Add_SelectionChanged({ Refresh-Preview })
$ctrls.Fa.Add_Click({ Refresh-Preview }); $ctrls.Pin.Add_Click({ Refresh-Preview })
$ctrls.Scan.Add_Click({ Scan-Models })
$ctrls.AdvancedToggle.Add_Click({ $ctrls.PreviewBox.Visibility = if ($ctrls.AdvancedToggle.IsChecked) { 'Visible' } else { 'Collapsed' } })

$ctrls.BrowseFolder.Add_Click({
  $dlg = New-Object System.Windows.Forms.FolderBrowserDialog
  if ($ctrls.Folder.Text) { $dlg.SelectedPath = $ctrls.Folder.Text }
  if ($dlg.ShowDialog() -eq 'OK') { $ctrls.Folder.Text = $dlg.SelectedPath; Scan-Models }
})
$ctrls.BrowseServer.Add_Click({
  $dlg = New-Object System.Windows.Forms.OpenFileDialog; $dlg.Filter = "llama-server.exe|llama-server.exe|Exe (*.exe)|*.exe"
  if ($dlg.ShowDialog() -eq 'OK') { $ctrls.Server.Text = $dlg.FileName }
})

$ctrls.Launch.Add_Click({
  if (-not (Test-Path $ctrls.Server.Text)) { Log "! server exe not found: $($ctrls.Server.Text)"; return }
  $m = Get-ModelPath
  if (-not $m) { Log "! no model selected (set a folder, Scan, pick one)"; return }
  if (-not (Test-Path $m)) { Log "! model not found: $m"; return }
  Refresh-Preview
  if ($ctrls.Pin.IsChecked) { $env:GGML_CUDA_REGISTER_HOST = '1' } else { Remove-Item Env:\GGML_CUDA_REGISTER_HOST -ErrorAction SilentlyContinue }
  $script:outFile = [System.IO.Path]::GetTempFileName(); $script:errFile = [System.IO.Path]::GetTempFileName(); $script:opened = $false
  Log "launching: $($ctrls.Cmd.Text)"
  try {
    $script:proc = Start-Process -FilePath $ctrls.Server.Text -ArgumentList (Build-Args) -PassThru `
      -RedirectStandardOutput $script:outFile -RedirectStandardError $script:errFile -WindowStyle Hidden
  } catch { Log "! failed to start: $_"; return }
  $ctrls.Launch.IsEnabled = $false; $ctrls.Stop.IsEnabled = $true
  $script:timer = New-Object System.Windows.Threading.DispatcherTimer
  $script:timer.Interval = [TimeSpan]::FromMilliseconds(600)
  $script:timer.Add_Tick({
    $tail = ''
    foreach ($fp in @($script:errFile, $script:outFile)) {
      if ($fp -and (Test-Path $fp)) { $tail += (Get-Content $fp -Tail 6 -ErrorAction SilentlyContinue) -join "`n" }
    }
    if ($tail) { $ctrls.Log.Text = "--- server output ---`n$tail" }
    if (-not $script:opened -and $tail -match 'listening|server is listening|HTTP server') {
      $script:opened = $true; Start-Process "http://localhost:$($ctrls.Port.Text)"; Log "opened browser at :$($ctrls.Port.Text)"
    }
    if ($script:proc -and $script:proc.HasExited) {
      $script:timer.Stop(); Log "server exited (code $($script:proc.ExitCode))"
      $ctrls.Launch.IsEnabled = $true; $ctrls.Stop.IsEnabled = $false
    }
  })
  $script:timer.Start()
})

$ctrls.Stop.Add_Click({
  if ($script:proc -and -not $script:proc.HasExited) { $script:proc.Kill(); Log "stopped." }
  if ($script:timer) { $script:timer.Stop() }
  $ctrls.Launch.IsEnabled = $true; $ctrls.Stop.IsEnabled = $false
})
$win.Add_Closed({ if ($script:proc -and -not $script:proc.HasExited) { $script:proc.Kill() } })

if ($ctrls.Folder.Text) { Scan-Models } else { Refresh-Preview }

if ($env:LAUNCHER_SELFTEST) {
  Write-Output "SELFTEST OK"
  Write-Output ("controls bound: " + $ctrls.Keys.Count)
  Write-Output ("models listed: " + @($ctrls.Model.ItemsSource).Count)
  Write-Output ("fresh default: preset=" + $ctrls.Preset.Text + " ncmoe='" + $ctrls.Ncmoe.Text + "' ub='" + $ctrls.Ub.Text + "' pin=" + $ctrls.Pin.IsChecked)
  Write-Output ("fresh preview: " + $ctrls.Cmd.Text)
  $ctrls.Preset.SelectedIndex = 0  # exercise the offload profile handler
  Write-Output ("offload profile: ncmoe='" + $ctrls.Ncmoe.Text + "' ub='" + $ctrls.Ub.Text + "' pin=" + $ctrls.Pin.IsChecked)
  return
}

$win.ShowDialog() | Out-Null
