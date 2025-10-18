#include <foundry.h>

#include "foundry-git-monitor-private.h"

static GMainLoop *main_loop;
static const char *dirpath;

static DexFuture *
main_fiber (gpointer data)
{
  g_autoptr(FoundryGitMonitor) monitor = NULL;
  g_autoptr(GError) error = NULL;

  dex_await (foundry_init (), NULL);

  monitor = dex_await_object (foundry_git_monitor_new (dirpath), &error);
  g_assert_no_error (error);
  g_assert_nonnull (monitor);

  g_print ("Ready...\n");

  for (;;)
    {
      gboolean r = dex_await (foundry_git_monitor_when_changed (monitor), &error);

      g_assert_no_error (error);
      g_assert_true (r);

      g_print ("Changed\n");
    }

  return NULL;
}

int
main (int   argc,
      char *argv[])
{
  if (argc != 2)
    {
      g_printerr ("usage: %s path/to/.git\n", argv[0]);
      return 1;
    }

  dirpath = argv[1];

  main_loop = g_main_loop_new (NULL, FALSE);
  dex_future_disown (dex_scheduler_spawn (NULL, 8*1024*1024, main_fiber, NULL, NULL));
  g_main_loop_run (main_loop);

  return 0;
}
