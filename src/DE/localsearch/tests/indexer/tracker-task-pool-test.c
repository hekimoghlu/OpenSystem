/*
 * Copyright (C) 2011, Nokia <ivan.frade@nokia.com>
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the
 * Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor,
 * Boston, MA  02110-1301, USA.
 *
 */
#include <glib.h>

/* NOTE: We're not including tracker-miner.h here because this is private. */
#include <tracker-task-pool.h>

static void
test_task_pool_limit_set (void)
{
        TrackerTaskPool *pool;

        pool = tracker_task_pool_new (5);
        g_assert_cmpint (tracker_task_pool_get_limit(pool), ==, 5);

        tracker_task_pool_set_limit (pool, 3);
        g_assert_cmpint (tracker_task_pool_get_limit(pool), ==, 3);

        g_assert_true (tracker_task_pool_limit_reached (pool) == FALSE);
        g_object_unref (pool);
}

static TrackerTask *
add_task (TrackerTaskPool *pool,
          const gchar     *filename,
          gint             expected_size,
          gboolean         hit_limit)
{
        TrackerTask *task;

        task = tracker_task_new (g_file_new_for_path (filename), NULL, NULL);
        tracker_task_pool_add (pool, task);

        g_assert_cmpint (tracker_task_pool_get_size (pool), ==, expected_size);
        g_assert_true (tracker_task_pool_limit_reached (pool) == hit_limit);

        return task;
}

static void
remove_task (TrackerTaskPool *pool,
             TrackerTask     *task,
             gint             expected_size,
             gboolean         hit_limit)
{
        tracker_task_pool_remove (pool, task);

        g_assert_cmpint (tracker_task_pool_get_size (pool), ==, expected_size);
        g_assert_true (hit_limit == tracker_task_pool_limit_reached (pool));

        g_object_unref (tracker_task_get_file (task));
        tracker_task_unref (task);
}

static void
test_task_pool_add_remove (void)
{
        TrackerTaskPool *pool;
        TrackerTask *a, *b, *c, *d, *nonexistent;

        pool = tracker_task_pool_new (3);

        /* Additions ... */
        a = add_task (pool, "/dev/null", 1, FALSE);
        b = add_task (pool, "/dev/null2", 2, FALSE);
        c = add_task (pool, "/dev/null3", 3, TRUE);

        /* We can go over the limit */
        d = add_task (pool, "/dev/null4", 4, TRUE);

        /* Remove something that doesn't exist */
        nonexistent = tracker_task_new (g_file_new_for_path ("/dev/null/imNotInThePool"), NULL, NULL);
        remove_task (pool, nonexistent, 4, TRUE);

        /* Removals ... (in different order)*/
        remove_task (pool, d, 3, TRUE);
        remove_task (pool, c, 2, FALSE);
        remove_task (pool, b, 1, FALSE);
        remove_task (pool, a, 0, FALSE);

        /* Remove in empty queue */
        nonexistent = tracker_task_new (g_file_new_for_path ("/dev/null/random"), NULL, NULL);
        remove_task (pool, nonexistent, 0, FALSE);

        g_object_unref (pool);
}

static void
test_task_pool_find (void)
{
        TrackerTaskPool *pool;
        gboolean task;
        GFile *goal;

        pool = tracker_task_pool_new (3);

        add_task (pool, "/dev/null", 1, FALSE);
        add_task (pool, "/dev/null2", 2, FALSE);
        add_task (pool, "/dev/null3", 3, TRUE);

        /* Search first, last, in the middle... */
        goal = g_file_new_for_path ("/dev/null2");
        task = tracker_task_pool_find (pool, goal);
        g_assert_true (task);
        g_object_unref (goal);

        goal = g_file_new_for_path ("/dev/null");
        task = tracker_task_pool_find (pool, goal);
        g_assert_true (task);
        g_object_unref (goal);

        goal = g_file_new_for_path ("/dev/null3");
        task = tracker_task_pool_find (pool, goal);
        g_assert_true (task);
        g_object_unref (goal);

        goal = g_file_new_for_path ("/dev/thisDoesntExists");
        task = tracker_task_pool_find (pool, goal);
        g_assert_true (task == FALSE);
        g_object_unref (goal);

        g_object_unref (pool);
}

static void
count_elements_cb (gpointer data,
                   gpointer user_data)
{
        gint *counter = (gint*)user_data;
        (*counter) += 1;
}

static void
test_task_pool_foreach (void)
{
        TrackerTaskPool *pool;
        int counter = 0;

        pool = tracker_task_pool_new (3);

        add_task (pool, "/dev/null", 1, FALSE);
        add_task (pool, "/dev/null2", 2, FALSE);
        add_task (pool, "/dev/null3", 3, TRUE);

        tracker_task_pool_foreach (pool, count_elements_cb, &counter);

        g_assert_cmpint (counter, ==, 3);
}

gint
main (gint argc, gchar **argv)
{
        g_test_init (&argc, &argv, NULL);

        g_test_add_func ("/libtracker-miner/tracker-task-pool/limit_set",
                         test_task_pool_limit_set);

        g_test_add_func ("/libtracker-miner/tracker-task-pool/add_remove",
                         test_task_pool_add_remove);

        g_test_add_func ("/libtracker-miner/tracker-task-pool/find",
                         test_task_pool_find);
 
        g_test_add_func ("/libtracker-miner/tracker-task-pool/foreach",
                         test_task_pool_foreach);

        return g_test_run ();
}
