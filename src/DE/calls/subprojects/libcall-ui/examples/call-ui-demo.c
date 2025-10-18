#include "cui-config.h"

#include <gtk/gtk.h>
#include <call-ui.h>

#include <glib/gi18n.h>

#include "cui-demo-window.h"

static void
startup (AdwApplication *app)
{
  cui_init (FALSE);
}

static void
show_window (AdwApplication *app)
{
  CuiDemoWindow *window;

  window = cui_demo_window_new (app);

  gtk_window_present (GTK_WINDOW (window));
}

int
main (int    argc,
      char **argv)
{
  AdwApplication *app;
  int status;

  /* This is enough since libcall-ui performs the bindtextdomain */
  textdomain (GETTEXT_PACKAGE);

  app = adw_application_new ("org.gnome.CallUI.Demo",
#if GLIB_CHECK_VERSION (2, 73, 3)
                             G_APPLICATION_DEFAULT_FLAGS);
#else
                             G_APPLICATION_FLAGS_NONE);
#endif
  g_signal_connect (app, "startup", G_CALLBACK (startup), NULL);
  g_signal_connect (app, "activate", G_CALLBACK (show_window), NULL);
  status = g_application_run (G_APPLICATION (app), argc, argv);
  g_object_unref (app);

  return status;
}
