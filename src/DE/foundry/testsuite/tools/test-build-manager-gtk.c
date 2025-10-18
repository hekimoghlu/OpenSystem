#include <foundry.h>
#include <gtk/gtk.h>
#include <vte/vte.h>

#include <glib/gstdio.h>

static GMainLoop *main_loop;
static const char *dirpath;
static FoundryPtyDiagnostics *diagnostics;

static void
setup_row (GtkSignalListItemFactory *factory,
           GtkListItem              *item,
           gpointer                  user_data)
{
  gtk_list_item_set_child (item,
                           g_object_new (GTK_TYPE_LABEL,
                                         "xalign", 0.f,
                                         NULL));
}

static void
bind_row (GtkSignalListItemFactory *factory,
          GtkListItem              *item,
          gpointer                  user_data)
{
  FoundryDiagnostic *diagnostic = gtk_list_item_get_item (item);
  g_autoptr(GFile) file = foundry_diagnostic_dup_file (diagnostic);
  GtkLabel *label = GTK_LABEL (gtk_list_item_get_child (item));
  g_autoptr(GString) str = g_string_new (NULL);
  g_autofree char *message = foundry_diagnostic_dup_message (diagnostic);
  guint line = foundry_diagnostic_get_line (diagnostic);
  guint line_offset = foundry_diagnostic_get_line_offset (diagnostic);

  g_string_append (str, g_file_peek_path (file));
  g_string_append_printf (str, ":%u", line+1);
  g_string_append_printf (str, ":%u", line_offset+1);

  if (message)
    g_string_append_printf (str, ": %s", message);

  gtk_label_set_label (label, str->str);
}

static DexFuture *
main_fiber (gpointer data)
{
  g_autoptr(GError) error = NULL;
  g_autoptr(FoundryBuildManager) build_manager = NULL;
  g_autoptr(FoundryRunManager) run_manager = NULL;
  g_autoptr(GtkListItemFactory) factory = NULL;
  g_autoptr(GtkSelectionModel) model = NULL;
  g_autoptr(FoundryContext) context = NULL;
  g_autoptr(VtePty) pty = NULL;
  g_autofree char *path = NULL;
  GActionGroup *action_group = NULL;
  GtkScrolledWindow *scroller;
  VteTerminal *terminal;
  GtkListView *listview;
  GtkWindow *window;
  GtkBox *hbox;
  GtkBox *box;
  g_autofd int pty_fd = -1;

  dex_await (foundry_init (), NULL);

  if (!(path = dex_await_string (foundry_context_discover (dirpath, NULL), &error)))
    g_error ("%s", error->message);

  if (!(context = dex_await_object (foundry_context_new (path, dirpath, FOUNDRY_CONTEXT_FLAGS_NONE, NULL), &error)))
    g_error ("%s", error->message);

  window = g_object_new (GTK_TYPE_WINDOW,
                         "default-width", 400,
                         "default-height", 600,
                         NULL);

  action_group = foundry_context_dup_action_group (context);
  gtk_widget_insert_action_group (GTK_WIDGET (window), "context", action_group);
  g_clear_object (&action_group);

  box = g_object_new (GTK_TYPE_BOX,
                      "orientation", GTK_ORIENTATION_VERTICAL,
                      NULL);
  gtk_window_set_child (window, GTK_WIDGET (box));

  hbox = g_object_new (GTK_TYPE_BOX,
                       "orientation", GTK_ORIENTATION_HORIZONTAL,
                       NULL);
  gtk_box_append (box, GTK_WIDGET (hbox));

  gtk_box_append (hbox,
                  g_object_new (GTK_TYPE_BUTTON,
                                "label", "Build",
                                "action-name", "context.build-manager.build",
                                NULL));
  gtk_box_append (hbox,
                  g_object_new (GTK_TYPE_BUTTON,
                                "label", "Clean",
                                "action-name", "context.build-manager.clean",
                                NULL));
  gtk_box_append (hbox,
                  g_object_new (GTK_TYPE_BUTTON,
                                "label", "Invalidate",
                                "action-name", "context.build-manager.invalidate",
                                NULL));
  gtk_box_append (hbox,
                  g_object_new (GTK_TYPE_BUTTON,
                                "label", "Purge",
                                "action-name", "context.build-manager.purge",
                                NULL));
  gtk_box_append (hbox,
                  g_object_new (GTK_TYPE_BUTTON,
                                "label", "Rebuild",
                                "action-name", "context.build-manager.rebuild",
                                NULL));
  gtk_box_append (hbox,
                  g_object_new (GTK_TYPE_BUTTON,
                                "label", "Run",
                                "action-name", "context.run-manager.run",
                                NULL));
  gtk_box_append (hbox,
                  g_object_new (GTK_TYPE_BUTTON,
                                "label", "Update Dependencies",
                                "action-name", "context.dependency-manager.update",
                                NULL));
  gtk_box_append (hbox,
                  g_object_new (GTK_TYPE_BUTTON,
                                "label", "Stop",
                                "action-name", "context.build-manager.stop",
                                NULL));

  scroller = g_object_new (GTK_TYPE_SCROLLED_WINDOW,
                           "vexpand", TRUE,
                           NULL);
  gtk_box_append (box, GTK_WIDGET (scroller));

  terminal = g_object_new (VTE_TYPE_TERMINAL, NULL);
  gtk_widget_set_size_request (GTK_WIDGET (terminal), 400, 200);
  gtk_box_append (box, GTK_WIDGET (terminal));

  pty = vte_pty_new_sync (VTE_PTY_DEFAULT, NULL, NULL);
  vte_terminal_set_pty (terminal, pty);

  diagnostics = foundry_pty_diagnostics_new (context, vte_pty_get_fd (pty));
  pty_fd = foundry_pty_diagnostics_create_producer (diagnostics, &error);

  build_manager = foundry_context_dup_build_manager (context);
  foundry_build_manager_set_default_pty (build_manager, pty_fd);

  run_manager = foundry_context_dup_run_manager (context);
  foundry_run_manager_set_default_pty (run_manager, pty_fd);

  factory = gtk_signal_list_item_factory_new ();
  g_signal_connect (factory, "setup", G_CALLBACK (setup_row), NULL);
  g_signal_connect (factory, "bind", G_CALLBACK (bind_row), NULL);

  model = GTK_SELECTION_MODEL (gtk_no_selection_new (g_object_ref (G_LIST_MODEL (diagnostics))));
  listview = g_object_new (GTK_TYPE_LIST_VIEW,
                           "height-request", 200,
                           "factory", factory,
                           "model", model,
                           NULL);
  gtk_scrolled_window_set_child (scroller, GTK_WIDGET (listview));

  g_signal_connect_swapped (window,
                            "close-request",
                            G_CALLBACK (g_main_loop_quit),
                            main_loop);
  gtk_window_present (window);

  return NULL;
}

int
main (int   argc,
      char *argv[])
{
  if (argc != 2)
    {
      g_printerr ("usage: %s PROJECT_DIR\n", argv[0]);
      return 1;
    }

  dirpath = argv[1];

  gtk_init ();

  main_loop = g_main_loop_new (NULL, FALSE);
  dex_future_disown (dex_scheduler_spawn (NULL, 8*1024*1024, main_fiber, NULL, NULL));
  g_main_loop_run (main_loop);

  return 0;
}
