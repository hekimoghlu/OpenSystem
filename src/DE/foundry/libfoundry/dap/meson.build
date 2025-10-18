libfoundry_config_conf.set10('FOUNDRY_FEATURE_DAP', true)

foundry_sources += files([
  'foundry-dap-debugger.c',
  'foundry-dap-protocol.c',
])

foundry_private_sources += files([
  'foundry-dap-debugger-breakpoint.c',
  'foundry-dap-debugger-log-message.c',
  'foundry-dap-debugger-instruction.c',
  'foundry-dap-debugger-module.c',
  'foundry-dap-debugger-source.c',
  'foundry-dap-debugger-stop-event.c',
  'foundry-dap-debugger-thread.c',
  'foundry-dap-debugger-stack-frame.c',
  'foundry-dap-debugger-variable.c',
  'foundry-dap-debugger-watchpoint.c',
  'foundry-dap-driver.c',
  'foundry-dap-waiter.c',
])

foundry_headers += files([
  'foundry-dap-debugger.h',
  'foundry-dap-protocol.h',
])

foundry_include_directories += [include_directories('.')]
