/*
 * Copyright (C) 2021, Red Hat Inc
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
 * Author: Carlos Garnacho <carlosg@gnome.org>
 */

#include "config-miners.h"

#include "tracker-monitor.h"
#include "tracker-monitor-private.h"

#include "tracker-monitor-glib.h"
#include "tracker-monitor-fanotify.h"

enum {
	ITEM_CREATED,
	ITEM_UPDATED,
	ITEM_ATTRIBUTE_UPDATED,
	ITEM_DELETED,
	ITEM_MOVED,
	LAST_SIGNAL
};

enum {
	PROP_0,
	PROP_ENABLED,
	PROP_LIMIT,
	PROP_COUNT,
	PROP_IGNORED,
	N_PROPS
};

static guint signals[LAST_SIGNAL] = { 0, };
static GParamSpec *pspecs[N_PROPS] = { 0, };

G_DEFINE_ABSTRACT_TYPE (TrackerMonitor, tracker_monitor, G_TYPE_OBJECT)

static void
tracker_monitor_set_property (GObject      *object,
                              guint         prop_id,
                              const GValue *value,
                              GParamSpec   *pspec)
{
	switch (prop_id) {
	case PROP_ENABLED:
	case PROP_LIMIT:
	case PROP_COUNT:
	case PROP_IGNORED:
		g_warning ("Property should be overridden by superclass");
		break;
	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
	}
}

static void
tracker_monitor_get_property (GObject      *object,
                              guint         prop_id,
                              GValue       *value,
                              GParamSpec   *pspec)
{
	switch (prop_id) {
	case PROP_ENABLED:
	case PROP_LIMIT:
	case PROP_COUNT:
	case PROP_IGNORED:
		g_warning ("Property should be overridden by superclass");
		break;
	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
	}
}

static void
tracker_monitor_class_init (TrackerMonitorClass *klass)
{
	GObjectClass *object_class;

	object_class = G_OBJECT_CLASS (klass);

	object_class->set_property = tracker_monitor_set_property;
	object_class->get_property = tracker_monitor_get_property;

	signals[ITEM_CREATED] =
		g_signal_new ("item-created",
		              G_TYPE_FROM_CLASS (klass),
		              G_SIGNAL_RUN_LAST,
		              0,
		              NULL, NULL,
		              NULL,
		              G_TYPE_NONE,
		              2,
		              G_TYPE_OBJECT,
		              G_TYPE_BOOLEAN);
	signals[ITEM_UPDATED] =
		g_signal_new ("item-updated",
		              G_TYPE_FROM_CLASS (klass),
		              G_SIGNAL_RUN_LAST,
		              0,
		              NULL, NULL,
		              NULL,
		              G_TYPE_NONE,
		              2,
		              G_TYPE_OBJECT,
		              G_TYPE_BOOLEAN);
	signals[ITEM_ATTRIBUTE_UPDATED] =
		g_signal_new ("item-attribute-updated",
		              G_TYPE_FROM_CLASS (klass),
		              G_SIGNAL_RUN_LAST,
		              0,
		              NULL, NULL,
		              NULL,
		              G_TYPE_NONE,
		              2,
		              G_TYPE_OBJECT,
		              G_TYPE_BOOLEAN);
	signals[ITEM_DELETED] =
		g_signal_new ("item-deleted",
		              G_TYPE_FROM_CLASS (klass),
		              G_SIGNAL_RUN_LAST,
		              0,
		              NULL, NULL,
		              NULL,
		              G_TYPE_NONE,
		              2,
		              G_TYPE_OBJECT,
		              G_TYPE_BOOLEAN);
	signals[ITEM_MOVED] =
		g_signal_new ("item-moved",
		              G_TYPE_FROM_CLASS (klass),
		              G_SIGNAL_RUN_LAST,
		              0,
		              NULL, NULL,
		              NULL,
		              G_TYPE_NONE,
		              4,
		              G_TYPE_OBJECT,
		              G_TYPE_OBJECT,
		              G_TYPE_BOOLEAN,
		              G_TYPE_BOOLEAN);

	pspecs[PROP_ENABLED] =
		g_param_spec_boolean ("enabled",
		                      "Enabled",
		                      "Enabled",
		                      TRUE,
		                      G_PARAM_READWRITE |
		                      G_PARAM_STATIC_STRINGS);
	pspecs[PROP_LIMIT] =
		g_param_spec_uint ("limit",
		                   "Limit",
		                   "Limit",
		                   0,
		                   G_MAXUINT,
		                   0,
		                   G_PARAM_READABLE |
		                   G_PARAM_STATIC_STRINGS);
	pspecs[PROP_COUNT] =
		g_param_spec_uint ("count",
		                   "Count",
		                   "Count",
		                   0,
		                   G_MAXUINT,
		                   0,
		                   G_PARAM_READABLE |
		                   G_PARAM_STATIC_STRINGS);
	pspecs[PROP_IGNORED] =
		g_param_spec_uint ("ignored",
		                   "Ignored",
		                   "Ignored",
		                   0,
		                   G_MAXUINT,
		                   0,
		                   G_PARAM_READABLE |
		                   G_PARAM_STATIC_STRINGS);

	g_object_class_install_properties (object_class, N_PROPS, pspecs);
}

static void
tracker_monitor_init (TrackerMonitor *object)
{
}

gboolean
tracker_monitor_move (TrackerMonitor *monitor,
                      GFile          *old_file,
                      GFile          *new_file)
{
	g_return_val_if_fail (TRACKER_IS_MONITOR (monitor), FALSE);
	g_return_val_if_fail (G_IS_FILE (old_file), FALSE);
	g_return_val_if_fail (G_IS_FILE (new_file), FALSE);

	return TRACKER_MONITOR_GET_CLASS (monitor)->move (monitor,
	                                                  old_file,
	                                                  new_file);
}

gboolean
tracker_monitor_get_enabled (TrackerMonitor *monitor)
{
	gboolean enabled;

	g_return_val_if_fail (TRACKER_IS_MONITOR (monitor), 0);

	g_object_get (monitor, "enabled", &enabled, NULL);

	return enabled;
}

void
tracker_monitor_set_enabled (TrackerMonitor *monitor,
                             gboolean        enabled)
{
	g_return_if_fail (TRACKER_IS_MONITOR (monitor));

	TRACKER_MONITOR_GET_CLASS (monitor)->set_enabled (monitor, !!enabled);
}

gboolean
tracker_monitor_add (TrackerMonitor *monitor,
                     GFile          *file)
{
	g_return_val_if_fail (TRACKER_IS_MONITOR (monitor), FALSE);
	g_return_val_if_fail (G_IS_FILE (file), FALSE);

	return TRACKER_MONITOR_GET_CLASS (monitor)->add (monitor, file);
}

gboolean
tracker_monitor_remove (TrackerMonitor *monitor,
                        GFile          *file)
{
	g_return_val_if_fail (TRACKER_IS_MONITOR (monitor), FALSE);
	g_return_val_if_fail (G_IS_FILE (file), FALSE);

	return TRACKER_MONITOR_GET_CLASS (monitor)->remove (monitor,
	                                                    file);
}

gboolean
tracker_monitor_remove_recursively (TrackerMonitor *monitor,
                                    GFile          *file)
{
	g_return_val_if_fail (TRACKER_IS_MONITOR (monitor), FALSE);
	g_return_val_if_fail (G_IS_FILE (file), FALSE);

	return TRACKER_MONITOR_GET_CLASS (monitor)->remove_recursively (monitor,
	                                                                file,
	                                                                FALSE);
}

gboolean
tracker_monitor_remove_children_recursively (TrackerMonitor *monitor,
                                             GFile          *file)
{
	g_return_val_if_fail (TRACKER_IS_MONITOR (monitor), FALSE);
	g_return_val_if_fail (G_IS_FILE (file), FALSE);

	return TRACKER_MONITOR_GET_CLASS (monitor)->remove_recursively (monitor,
	                                                                file,
	                                                                TRUE);
}

gboolean
tracker_monitor_is_watched (TrackerMonitor *monitor,
                            GFile          *file)
{
	g_return_val_if_fail (TRACKER_IS_MONITOR (monitor), FALSE);
	g_return_val_if_fail (G_IS_FILE (file), FALSE);

	return TRACKER_MONITOR_GET_CLASS (monitor)->is_watched (monitor,
	                                                        file);
}

guint
tracker_monitor_get_count (TrackerMonitor *monitor)
{
	g_return_val_if_fail (TRACKER_IS_MONITOR (monitor), 0);

	return TRACKER_MONITOR_GET_CLASS (monitor)->get_count (monitor);
}

guint
tracker_monitor_get_ignored (TrackerMonitor *monitor)
{
	guint ignored;

	g_return_val_if_fail (TRACKER_IS_MONITOR (monitor), 0);

	g_object_get (monitor, "ignored", &ignored, NULL);

	return ignored;
}

guint
tracker_monitor_get_limit (TrackerMonitor *monitor)
{
	guint limit;

	g_return_val_if_fail (TRACKER_IS_MONITOR (monitor), 0);

	g_object_get (monitor, "limit", &limit, NULL);

	return limit;
}

void
tracker_monitor_emit_created (TrackerMonitor *monitor,
                              GFile          *file,
                              gboolean        is_directory)

{
	g_signal_emit (monitor,
	               signals[ITEM_CREATED], 0,
	               file, is_directory);
}

void
tracker_monitor_emit_updated (TrackerMonitor *monitor,
                              GFile          *file,
                              gboolean        is_directory)
{
	g_signal_emit (monitor,
	               signals[ITEM_UPDATED], 0,
	               file, is_directory);
}

void
tracker_monitor_emit_attributes_updated (TrackerMonitor *monitor,
                                         GFile          *file,
                                         gboolean        is_directory)
{
	g_signal_emit (monitor,
	               signals[ITEM_ATTRIBUTE_UPDATED], 0,
	               file, is_directory);
}

void
tracker_monitor_emit_deleted (TrackerMonitor *monitor,
                              GFile          *file,
                              gboolean        is_directory)
{
	g_signal_emit (monitor,
	               signals[ITEM_DELETED], 0,
	               file, is_directory);
}

void
tracker_monitor_emit_moved (TrackerMonitor *monitor,
                            GFile          *file,
                            GFile          *other_file,
                            gboolean        is_directory)
{
	g_signal_emit (monitor,
	               signals[ITEM_MOVED], 0,
	               file, other_file,
	               is_directory, TRUE);
}

TrackerMonitor *
tracker_monitor_new (GError **error)
{
#ifdef HAVE_FANOTIFY
	TrackerMonitor *monitor;

	monitor = g_initable_new (TRACKER_TYPE_MONITOR_FANOTIFY,
	                          NULL, NULL, NULL);
	if (monitor)
		return monitor;
#endif

	return g_initable_new (TRACKER_TYPE_MONITOR_GLIB,
	                       NULL, error, NULL);
}
