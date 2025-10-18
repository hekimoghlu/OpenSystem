#include <foundry.h>
#include <gtk/gtk.h>

static FoundryLlmConversation *conversation;
static GListModel *all_tools;
static GMainLoop *main_loop;
static const char *model_name;

static void
entry_activate (GtkEntry *entry)
{
  g_autofree char *text = NULL;

  g_assert (GTK_IS_ENTRY (entry));

  text = g_strdup (gtk_editable_get_text (GTK_EDITABLE (entry)));

  if (text[0] == 0)
    return;

  gtk_editable_set_text (GTK_EDITABLE (entry), "");

  foundry_llm_conversation_send_message (conversation, "user", text);
}

static void
setup_row (GtkSignalListItemFactory *factory,
           GtkListItem              *item,
           gpointer                  user_data)
{
  GtkBox *box;
  GtkLabel *role;
  GtkLabel *content;

  box = g_object_new (GTK_TYPE_BOX,
                      "orientation", GTK_ORIENTATION_HORIZONTAL,
                      NULL);
  role = g_object_new (GTK_TYPE_LABEL,
                       "width-request", 75,
                       "xalign", 0.f,
                       "yalign", 0.f,
                       NULL);
  content = g_object_new (GTK_TYPE_LABEL,
                          "hexpand", TRUE,
                          "xalign", 0.f,
                          "yalign", 0.f,
                          "wrap", TRUE,
                          "wrap-mode", GTK_WRAP_CHAR,
                          NULL);
  gtk_box_append (box, GTK_WIDGET (role));
  gtk_box_append (box, GTK_WIDGET (content));

  gtk_list_item_set_child (item, GTK_WIDGET (box));
}

static void
bind_row (GtkSignalListItemFactory *factory,
          GtkListItem              *item,
          gpointer                  user_data)
{
  FoundryLlmMessage *message = gtk_list_item_get_item (item);
  g_autofree char *role = foundry_llm_message_dup_role (message);
  g_autofree char *content = foundry_llm_message_dup_content (message);
  GtkWidget *box = gtk_list_item_get_child (item);

  if (foundry_llm_message_has_tool_call (message))
    g_printerr ("TODO: Show row for tool call instead\n");

  gtk_label_set_label (GTK_LABEL (gtk_widget_get_first_child (box)), role);
  g_object_bind_property (message, "content",
                          gtk_widget_get_last_child (box), "label",
                          G_BINDING_SYNC_CREATE);
}

static void
add_context_cb (GtkButton   *button,
                GtkTextView *text)
{
  GtkTextBuffer *buffer = gtk_text_view_get_buffer (text);
  g_autofree char *str = NULL;
  GtkTextIter begin, end;

  gtk_text_buffer_get_bounds (buffer, &begin, &end);
  if (gtk_text_iter_equal (&begin, &end))
    return;

  str = gtk_text_iter_get_slice (&begin, &end);
  gtk_text_buffer_set_text (buffer, "", 0);

  foundry_llm_conversation_add_context (conversation, str);

  gtk_popover_popdown (GTK_POPOVER (gtk_widget_get_ancestor (GTK_WIDGET (text), GTK_TYPE_POPOVER)));
}

static void
notify_use_tools_cb (GtkToggleButton *button)
{
  if (gtk_toggle_button_get_active (button))
    foundry_llm_conversation_set_tools (conversation, all_tools);
  else
    foundry_llm_conversation_set_tools (conversation, NULL);
}

static DexFuture *
main_fiber (gpointer data)
{
  g_autoptr(GError) error = NULL;
  g_autoptr(GtkListItemFactory) factory = NULL;
  g_autoptr(GtkSelectionModel) model = NULL;
  g_autoptr(FoundryContext) context = NULL;
  g_autoptr(FoundryLlmManager) llm_manager = NULL;
  g_autoptr(FoundryLlmModel) llm = NULL;
  g_autoptr(GListModel) history = NULL;
  GtkScrolledWindow *scroller;
  g_autofree char *path = NULL;
  GtkToggleButton *tools_button;
  GtkMenuButton *menu_button;
  GtkHeaderBar *header;
  GtkListView *listview;
  GtkTextView *text;
  const char *dirpath;
  GtkPopover *popover;
  GtkButton *add;
  GtkWindow *window;
  GtkEntry *entry;
  GtkBox *box;
  GtkBox *pbox;

  dex_await (foundry_init (), NULL);

  dirpath = ".";

  if (!(path = dex_await_string (foundry_context_discover (dirpath, NULL), &error)))
    g_error ("%s", error->message);

  if (!(context = dex_await_object (foundry_context_new (path, dirpath, FOUNDRY_CONTEXT_FLAGS_NONE, NULL), &error)))
    g_error ("%s", error->message);

  window = g_object_new (GTK_TYPE_WINDOW,
                         "default-width", 400,
                         "default-height", 600,
                         NULL);

  box = g_object_new (GTK_TYPE_BOX,
                      "orientation", GTK_ORIENTATION_VERTICAL,
                      NULL);
  gtk_window_set_child (window, GTK_WIDGET (box));

  header = g_object_new (GTK_TYPE_HEADER_BAR, NULL);
  gtk_window_set_titlebar (window, GTK_WIDGET (header));

  tools_button = g_object_new (GTK_TYPE_TOGGLE_BUTTON,
                               "active", FALSE,
                               "label", "Use Tools",
                               "tooltip-text", "Enable use of registered tools (might disable streaming mode)",
                               NULL);
  g_signal_connect (tools_button,
                    "notify::active",
                    G_CALLBACK (notify_use_tools_cb),
                    NULL);
  gtk_header_bar_pack_start (header, GTK_WIDGET (tools_button));

  popover = g_object_new (GTK_TYPE_POPOVER, NULL);
  pbox = g_object_new (GTK_TYPE_BOX,
                       "orientation", GTK_ORIENTATION_VERTICAL,
                       NULL);
  gtk_popover_set_child (popover, GTK_WIDGET (pbox));
  scroller = g_object_new (GTK_TYPE_SCROLLED_WINDOW,
                           "width-request", 400,
                           "height-request", 400,
                           "vexpand", TRUE,
                           NULL);
  gtk_box_append (pbox, GTK_WIDGET (scroller));

  text = g_object_new (GTK_TYPE_TEXT_VIEW,
                       "monospace", TRUE,
                       "wrap-mode", GTK_WRAP_CHAR,
                       NULL);
  gtk_scrolled_window_set_child (scroller, GTK_WIDGET (text));

  add = g_object_new (GTK_TYPE_BUTTON,
                      "label", "Add Context",
                      NULL);
  g_signal_connect (add, "clicked", G_CALLBACK (add_context_cb), text);
  gtk_box_append (pbox, GTK_WIDGET (add));

  menu_button = g_object_new (GTK_TYPE_MENU_BUTTON,
                              "label", "Add Context",
                              "popover", popover,
                              NULL);
  gtk_header_bar_pack_start (header, GTK_WIDGET (menu_button));

  scroller = g_object_new (GTK_TYPE_SCROLLED_WINDOW,
                           "vexpand", TRUE,
                           NULL);
  gtk_box_append (box, GTK_WIDGET (scroller));

  entry = g_object_new (GTK_TYPE_ENTRY,
                        "margin-top", 6,
                        "margin-start", 6,
                        "margin-end", 6,
                        "margin-bottom", 6,
                        NULL);
  g_signal_connect (entry, "activate", G_CALLBACK (entry_activate), NULL);
  gtk_box_append (box, GTK_WIDGET (entry));

  llm_manager = foundry_context_dup_llm_manager (context);
  all_tools = dex_await_object (foundry_llm_manager_list_tools (llm_manager), NULL);

  if (!(llm = dex_await_object (foundry_llm_manager_find_model (llm_manager, model_name), &error)))
    g_error ("%s", error->message);

  if (!(conversation = dex_await_object (foundry_llm_model_chat (llm, "You are an assistant. You help find issues with the codebase and offer to build or fix other problems using tools at your disposal."), &error)))
    g_error ("%s", error->message);

  factory = gtk_signal_list_item_factory_new ();
  g_signal_connect (factory, "setup", G_CALLBACK (setup_row), NULL);
  g_signal_connect (factory, "bind", G_CALLBACK (bind_row), NULL);

  history = foundry_llm_conversation_list_history (conversation);
  model = GTK_SELECTION_MODEL (gtk_no_selection_new (g_object_ref (history)));
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
  gtk_widget_grab_focus (GTK_WIDGET (entry));

  return NULL;
}

int
main (int   argc,
      char *argv[])
{
  if (argc != 2)
    {
      g_printerr ("usage: %s MODEL_NAME\n", argv[0]);
      return 1;
    }

  model_name = argv[1];

  gtk_init ();

  main_loop = g_main_loop_new (NULL, FALSE);
  dex_future_disown (dex_scheduler_spawn (NULL, 0, main_fiber, NULL, NULL));
  g_main_loop_run (main_loop);

  return 0;
}
