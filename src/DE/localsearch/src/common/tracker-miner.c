/*
 * Copyright (C) 2009, Nokia <ivan.frade@nokia.com>
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
 */

#include "config-miners.h"

#include <math.h>

#include <glib/gi18n.h>

#include "tracker-dbus.h"
#include "tracker-debug.h"
#include "tracker-type-utils.h"

#include "tracker-miner.h"

/* Here we use ceil() to eliminate decimal points beyond what we're
 * interested in, which is 2 decimal places for the progress. The
 * ceil() call will also round up the last decimal place.
 *
 * The 0.49 value is used for rounding correctness, because ceil()
 * rounds up if the number is > 0.0.
 */
#define PROGRESS_ROUNDED(x) ((x) < 0.01 ? 0.00 : (ceil (((x) * 100) - 0.49) / 100))

/**
 * SECTION:tracker-miner-object
 * @short_description: Abstract base class for data miners
 * @include: libtracker-miner/tracker-miner.h
 *
 * #TrackerMiner is an abstract base class to help developing data miners
 * for tracker-store, being an abstract class it doesn't do much by itself,
 * but provides the basic signaling and control over the actual indexing
 * task.
 **/

typedef struct _TrackerMinerPrivate TrackerMinerPrivate;

struct _TrackerMinerPrivate {
	TrackerSparqlConnection *connection;
	gboolean started;
	gint n_pauses;
	gchar *status;
	gdouble progress;
	gint remaining_time;
	guint update_id;
};

enum {
	PROP_0,
	PROP_STATUS,
	PROP_PROGRESS,
	PROP_REMAINING_TIME,
	PROP_CONNECTION,
	N_PROPS
};

static GParamSpec *props[N_PROPS] = { 0, };

enum {
	STARTED,
	STOPPED,
	PAUSED,
	RESUMED,
	PROGRESS,
	LAST_SIGNAL
};

static guint signals[LAST_SIGNAL] = { 0 };

static void       miner_set_property           (GObject                *object,
                                                guint                   param_id,
                                                const GValue           *value,
                                                GParamSpec             *pspec);
static void       miner_get_property           (GObject                *object,
                                                guint                   param_id,
                                                GValue                 *value,
                                                GParamSpec             *pspec);
static void       miner_finalize               (GObject                *object);

/**
 * tracker_miner_error_quark:
 *
 * Gives the caller the #GQuark used to identify #TrackerMiner errors
 * in #GError structures. The #GQuark is used as the domain for the error.
 *
 * Returns: the #GQuark used for the domain of a #GError.
 *
 * Since: 0.8
 **/
G_DEFINE_QUARK (TrackerMinerError, tracker_miner_error)

G_DEFINE_ABSTRACT_TYPE_WITH_PRIVATE (TrackerMiner, tracker_miner, G_TYPE_OBJECT)

static void
tracker_miner_class_init (TrackerMinerClass *klass)
{
	GObjectClass *object_class = G_OBJECT_CLASS (klass);

	object_class->set_property = miner_set_property;
	object_class->get_property = miner_get_property;
	object_class->finalize     = miner_finalize;

	/**
	 * TrackerMiner::started:
	 * @miner: the #TrackerMiner
	 *
	 * the ::started signal is emitted in the miner
	 * right after it has been started through
	 * tracker_miner_start().
	 *
	 * Since: 0.8
	 **/
	signals[STARTED] =
		g_signal_new ("started",
		              G_OBJECT_CLASS_TYPE (object_class),
		              G_SIGNAL_RUN_LAST,
		              G_STRUCT_OFFSET (TrackerMinerClass, started),
		              NULL, NULL,
		              NULL,
		              G_TYPE_NONE, 0);
	/**
	 * TrackerMiner::stopped:
	 * @miner: the #TrackerMiner
	 *
	 * the ::stopped signal is emitted in the miner
	 * right after it has been stopped through
	 * tracker_miner_stop().
	 *
	 * Since: 0.8
	 **/
	signals[STOPPED] =
		g_signal_new ("stopped",
		              G_OBJECT_CLASS_TYPE (object_class),
		              G_SIGNAL_RUN_LAST,
		              G_STRUCT_OFFSET (TrackerMinerClass, stopped),
		              NULL, NULL,
		              NULL,
		              G_TYPE_NONE, 0);
	/**
	 * TrackerMiner::paused:
	 * @miner: the #TrackerMiner
	 *
	 * the ::paused signal is emitted whenever
	 * there is any reason to pause, either
	 * internal (through tracker_miner_pause()) or
	 * external (through DBus, see #TrackerMinerManager).
	 *
	 * Since: 0.8
	 **/
	signals[PAUSED] =
		g_signal_new ("paused",
		              G_OBJECT_CLASS_TYPE (object_class),
		              G_SIGNAL_RUN_LAST,
		              G_STRUCT_OFFSET (TrackerMinerClass, paused),
		              NULL, NULL,
		              NULL,
		              G_TYPE_NONE, 0);
	/**
	 * TrackerMiner::resumed:
	 * @miner: the #TrackerMiner
	 *
	 * the ::resumed signal is emitted whenever
	 * all reasons to pause have disappeared, see
	 * tracker_miner_resume() and #TrackerMinerManager.
	 *
	 * Since: 0.8
	 **/
	signals[RESUMED] =
		g_signal_new ("resumed",
		              G_OBJECT_CLASS_TYPE (object_class),
		              G_SIGNAL_RUN_LAST,
		              G_STRUCT_OFFSET (TrackerMinerClass, resumed),
		              NULL, NULL,
		              NULL,
		              G_TYPE_NONE, 0);
	/**
	 * TrackerMiner::progress:
	 * @miner: the #TrackerMiner
	 * @status: miner status
	 * @progress: a #gdouble indicating miner progress, from 0 to 1.
	 * @remaining_time: a #gint indicating the reamaining processing time, in
	 * seconds.
	 *
	 * the ::progress signal will be emitted by TrackerMiner implementations
	 * to indicate progress about the data mining process. @status will
	 * contain a translated string with the current miner status and @progress
	 * will indicate how much has been processed so far. @remaining_time will
	 * give the number expected of seconds to finish processing, 0 if the
	 * value cannot be estimated, and -1 if its not applicable.
	 *
	 * Since: 0.12
	 **/
	signals[PROGRESS] =
		g_signal_new ("progress",
		              G_OBJECT_CLASS_TYPE (object_class),
		              G_SIGNAL_RUN_LAST,
		              G_STRUCT_OFFSET (TrackerMinerClass, progress),
		              NULL, NULL,
		              NULL,
		              G_TYPE_NONE, 3,
		              G_TYPE_STRING,
		              G_TYPE_DOUBLE,
		              G_TYPE_INT);

	props[PROP_STATUS] =
		g_param_spec_string ("status", NULL, NULL,
		                     NULL,
		                     G_PARAM_READWRITE |
		                     G_PARAM_CONSTRUCT |
		                     G_PARAM_STATIC_STRINGS);
	props[PROP_PROGRESS] =
		g_param_spec_double ("progress", NULL, NULL,
		                     0.0, 1.0, 0.0,
		                     G_PARAM_READWRITE |
		                     G_PARAM_CONSTRUCT |
		                     G_PARAM_STATIC_STRINGS);
	props[PROP_REMAINING_TIME] =
		g_param_spec_int ("remaining-time", NULL, NULL,
		                  -1, G_MAXINT, -1,
		                  G_PARAM_READWRITE |
		                  G_PARAM_CONSTRUCT |
		                  G_PARAM_STATIC_STRINGS);
	props[PROP_CONNECTION] =
		g_param_spec_object ("connection", NULL, NULL,
		                     TRACKER_SPARQL_TYPE_CONNECTION,
		                     G_PARAM_READWRITE |
		                     G_PARAM_CONSTRUCT_ONLY |
		                     G_PARAM_STATIC_STRINGS);

	g_object_class_install_properties (object_class, N_PROPS, props);
}

static void
tracker_miner_init (TrackerMiner *miner)
{
}

static gboolean
miner_update_progress_cb (gpointer data)
{
	TrackerMiner *miner = data;
	TrackerMinerPrivate *priv = tracker_miner_get_instance_private (miner);

	g_signal_emit (miner, signals[PROGRESS], 0,
	               priv->status,
	               priv->progress,
	               priv->remaining_time);

	priv->update_id = 0;

	return FALSE;
}

static void
miner_set_property (GObject      *object,
                    guint         prop_id,
                    const GValue *value,
                    GParamSpec   *pspec)
{
	TrackerMiner *miner = TRACKER_MINER (object);
	TrackerMinerPrivate *priv = tracker_miner_get_instance_private (miner);

	switch (prop_id) {
	case PROP_STATUS: {
		const gchar *new_status;

		new_status = g_value_get_string (value);

		TRACKER_NOTE (STATUS,
		              g_message ("(Miner:'%s') set property:'status' to '%s'",
		                         G_OBJECT_TYPE_NAME (miner),
		                         new_status));

		if (priv->status && new_status &&
		    strcmp (priv->status, new_status) == 0) {
			/* Same, do nothing */
			break;
		}

		g_free (priv->status);
		priv->status = g_strdup (new_status);

		if (priv->update_id == 0) {
			priv->update_id = g_idle_add_full (G_PRIORITY_HIGH_IDLE,
			                                          miner_update_progress_cb,
			                                          miner,
			                                          NULL);
		}

		break;
	}
	case PROP_PROGRESS: {
		gdouble new_progress;

		new_progress = PROGRESS_ROUNDED (g_value_get_double (value));
		TRACKER_NOTE (STATUS,
		              g_message ("(Miner:'%s') Set property:'progress' to '%2.2f' (%2.2f before rounded)",
		                         G_OBJECT_TYPE_NAME (miner),
		                         new_progress,
		                         g_value_get_double (value)));

		/* NOTE: We don't round the current progress before
		 * comparison because we use the rounded value when
		 * we set it last.
		 *
		 * Only notify 1% changes
		 */
		if (new_progress == priv->progress) {
			/* Same, do nothing */
			break;
		}

		priv->progress = new_progress;

		if (priv->update_id == 0) {
			priv->update_id = g_idle_add_full (G_PRIORITY_HIGH_IDLE,
			                                   miner_update_progress_cb,
			                                   miner,
			                                   NULL);
		}

		break;
	}
	case PROP_REMAINING_TIME: {
		gint new_remaining_time;

		new_remaining_time = g_value_get_int (value);
		if (new_remaining_time != priv->remaining_time) {
			/* Just set the new remaining time, don't notify it */
			priv->remaining_time = new_remaining_time;
		}
		break;
	}
	case PROP_CONNECTION: {
		priv->connection = g_value_dup_object (value);
		break;
	}
	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
		break;
	}
}

static void
miner_get_property (GObject    *object,
                    guint       prop_id,
                    GValue     *value,
                    GParamSpec *pspec)
{
	TrackerMiner *miner = TRACKER_MINER (object);
	TrackerMinerPrivate *priv = tracker_miner_get_instance_private (miner);

	switch (prop_id) {
	case PROP_STATUS:
		g_value_set_string (value, priv->status);
		break;
	case PROP_PROGRESS:
		g_value_set_double (value, priv->progress);
		break;
	case PROP_REMAINING_TIME:
		g_value_set_int (value, priv->remaining_time);
		break;
	case PROP_CONNECTION:
		g_value_set_object (value, priv->connection);
		break;
	default:
		G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
		break;
	}
}

/**
 * tracker_miner_start:
 * @miner: a #TrackerMiner
 *
 * Tells the miner to start processing data.
 *
 * Since: 0.8
 **/
void
tracker_miner_start (TrackerMiner *miner)
{
	TrackerMinerPrivate *priv = tracker_miner_get_instance_private (miner);

	g_return_if_fail (TRACKER_IS_MINER (miner));

	if (priv->started == FALSE) {
		priv->started = TRUE;
		g_signal_emit (miner, signals[STARTED], 0);
	}
}

/**
 * tracker_miner_stop:
 * @miner: a #TrackerMiner
 *
 * Tells the miner to stop processing data.
 *
 * Since: 0.8
 **/
void
tracker_miner_stop (TrackerMiner *miner)
{
	TrackerMinerPrivate *priv = tracker_miner_get_instance_private (miner);

	g_return_if_fail (TRACKER_IS_MINER (miner));

	if (priv->started == TRUE) {
		priv->started = FALSE;
		g_signal_emit (miner, signals[STOPPED], 0);
	}
}

/**
 * tracker_miner_is_started:
 * @miner: a #TrackerMiner
 *
 * Returns #TRUE if the miner has been started.
 *
 * Returns: #TRUE if the miner is already started.
 *
 * Since: 0.8
 **/
gboolean
tracker_miner_is_started (TrackerMiner *miner)
{
	TrackerMinerPrivate *priv = tracker_miner_get_instance_private (miner);

	g_return_val_if_fail (TRACKER_IS_MINER (miner), TRUE);

	return priv->started;
}

/**
 * tracker_miner_is_paused:
 * @miner: a #TrackerMiner
 *
 * Returns #TRUE if the miner is paused.
 *
 * Returns: #TRUE if the miner is paused.
 *
 * Since: 0.10
 **/
gboolean
tracker_miner_is_paused (TrackerMiner *miner)
{
	TrackerMinerPrivate *priv = tracker_miner_get_instance_private (miner);

	g_return_val_if_fail (TRACKER_IS_MINER (miner), TRUE);

	return priv->n_pauses > 0;
}

/**
 * tracker_miner_pause:
 * @miner: a #TrackerMiner
 *
 * Asks @miner to pause. This call may be called multiple times,
 * but #TrackerMiner::paused will only be emitted the first time.
 * The same number of tracker_miner_resume() calls are expected
 * in order to resume operations.
 **/
void
tracker_miner_pause (TrackerMiner *miner)
{
	TrackerMinerPrivate *priv = tracker_miner_get_instance_private (miner);
	gint previous;

	g_return_if_fail (TRACKER_IS_MINER (miner));

	previous = g_atomic_int_add (&priv->n_pauses, 1);

	if (previous == 0)
		g_signal_emit (miner, signals[PAUSED], 0);
}

/**
 * tracker_miner_resume:
 * @miner: a #TrackerMiner
 *
 * Asks the miner to resume processing. This needs to be called
 * as many times as tracker_miner_pause() calls were done
 * previously. This function will return #TRUE when the miner
 * is actually resumed.
 *
 * Returns: #TRUE if the miner resumed its operations.
 **/
gboolean
tracker_miner_resume (TrackerMiner *miner)
{
	TrackerMinerPrivate *priv = tracker_miner_get_instance_private (miner);

	g_return_val_if_fail (TRACKER_IS_MINER (miner), FALSE);
	g_return_val_if_fail (priv->n_pauses > 0, FALSE);

	if (g_atomic_int_dec_and_test (&priv->n_pauses)) {
		g_signal_emit (miner, signals[RESUMED], 0);
		return TRUE;
	}

	return FALSE;
}

/**
 * tracker_miner_get_connection:
 * @miner: a #TrackerMiner
 *
 * Gets the #TrackerSparqlConnection initialized by @miner
 *
 * Returns: (transfer none): a #TrackerSparqlConnection.
 *
 * Since: 0.10
 **/
TrackerSparqlConnection *
tracker_miner_get_connection (TrackerMiner *miner)
{
	TrackerMinerPrivate *priv = tracker_miner_get_instance_private (miner);

	return priv->connection;
}

static void
miner_finalize (GObject *object)
{
	TrackerMiner *miner = TRACKER_MINER (object);
	TrackerMinerPrivate *priv = tracker_miner_get_instance_private (miner);

	if (priv->update_id != 0) {
		g_source_remove (priv->update_id);
	}

	g_free (priv->status);

	if (priv->connection) {
		g_object_unref (priv->connection);
	}

	G_OBJECT_CLASS (tracker_miner_parent_class)->finalize (object);
}
