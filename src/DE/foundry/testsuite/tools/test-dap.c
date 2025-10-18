#include <foundry.h>

#include "foundry-dap-driver-private.h"

static gboolean
on_request (FoundryDapDriver *driver,
            JsonNode         *message)
{
  const char *command = NULL;

  if (!FOUNDRY_JSON_OBJECT_PARSE (message, "command", FOUNDRY_JSON_NODE_GET_STRING (&command)))
    command = NULL;

  g_print ("Got request [%s]\n", command ? command : "--");

  return FALSE;
}

static void
on_event (FoundryDapDriver *driver,
          JsonNode         *message)
{
  const char *event = NULL;
  const char *category = NULL;
  const char *output = NULL;

  if (!FOUNDRY_JSON_OBJECT_PARSE (message, "event", FOUNDRY_JSON_NODE_GET_STRING (&event)))
    event = NULL;

  if (g_strcmp0 (event, "output") == 0 &&
      FOUNDRY_JSON_OBJECT_PARSE (message,
                                 "body", "{",
                                   "category", FOUNDRY_JSON_NODE_GET_STRING (&category),
                                   "output", FOUNDRY_JSON_NODE_GET_STRING (&output),
                                 "}"))
    g_print ("%s: %s", category, output);
  else
    g_print ("Got event [%s]\n", event ? event : "--");
}

int
main (int argc,
      char *argv[])
{
  g_autoptr(GMainLoop) main_loop = g_main_loop_new (NULL, FALSE);
  g_autoptr(GSubprocessLauncher) launcher = NULL;
  g_autoptr(FoundryDapDriver) driver = NULL;
  g_autoptr(GSubprocess) subprocess = NULL;
  g_autoptr(GIOStream) stream = NULL;
  g_autoptr(GError) error = NULL;

  dex_init ();
  dex_future_disown (foundry_init ());

  launcher = g_subprocess_launcher_new (G_SUBPROCESS_FLAGS_STDOUT_PIPE | G_SUBPROCESS_FLAGS_STDIN_PIPE);

  if (!(subprocess = g_subprocess_launcher_spawn (launcher, &error, "gdb", "--interpreter=dap", NULL)))
    g_error ("%s", error->message);

  stream = g_simple_io_stream_new (g_subprocess_get_stdout_pipe (subprocess),
                                   g_subprocess_get_stdin_pipe (subprocess));

  driver = foundry_dap_driver_new (stream, FOUNDRY_JSONRPC_STYLE_HTTP);

  g_signal_connect (driver,
                    "handle-request",
                    G_CALLBACK (on_request),
                    NULL);

  g_signal_connect (driver,
                    "event",
                    G_CALLBACK (on_event),
                    NULL);

  foundry_dap_driver_start (driver);

  g_main_loop_run (main_loop);

  return 0;
}
