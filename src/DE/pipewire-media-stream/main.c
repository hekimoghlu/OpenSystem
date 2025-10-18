#include <adwaita.h>
#include <glib/gi18n.h>
#include <pipewire/pipewire.h>

#include "window.h"

static void
activate (GtkApplication        *app,
          PmsWindowInitialState  initial)
{
  static GtkWindow *window = NULL;

  if (!window)
    window = g_object_new (PMS_TYPE_WINDOW,
                           "application", app,
                           "initial-state", initial,
                           NULL);

  gtk_window_present (window);
}

static void
on_activate (GtkApplication *app)
{
  activate (app, PMS_WINDOW_INITIAL_STATE_EMPTY);
}

static int
command_line (GApplication            *app,
              GApplicationCommandLine *cmdline)
{
  GVariantDict *options;
  gboolean screencast = FALSE;
  gboolean camera = FALSE;
  PmsWindowInitialState initial;

  options = g_application_command_line_get_options_dict (cmdline);
  g_variant_dict_lookup (options, "screencast", "b", &screencast);
  g_variant_dict_lookup (options, "camera", "b", &camera);

  if (screencast)
    initial = PMS_WINDOW_INITIAL_STATE_SCREENCAST;
  else if (camera)
    initial = PMS_WINDOW_INITIAL_STATE_CAMERA;
  else
    initial = PMS_WINDOW_INITIAL_STATE_EMPTY;

  activate (GTK_APPLICATION (app), initial);

  return 0;
}

int
main (int   argc,
      char *argv[])
{
  g_autoptr(AdwApplication) app = NULL;

  pw_init (&argc, &argv);

  app = adw_application_new ("com.feaneron.example.PipeWireMediaStream", G_APPLICATION_HANDLES_COMMAND_LINE);
  g_signal_connect (app, "activate", G_CALLBACK (on_activate), NULL);

  g_application_add_main_option (G_APPLICATION (app), "screencast", 0, 0, G_OPTION_ARG_NONE, "Show screencast", NULL);
  g_application_add_main_option (G_APPLICATION (app), "camera", 0, 0, G_OPTION_ARG_NONE, "Show camera stream", NULL);

  g_signal_connect (app, "command-line", G_CALLBACK (command_line), NULL);

  return g_application_run (G_APPLICATION (app), argc, argv);
}
