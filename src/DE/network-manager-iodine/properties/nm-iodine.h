/* -*- Mode: C; tab-width: 4; indent-tabs-mode: t; c-basic-offset: 4 -*- */
/*
 * GNOME UI dialogs for configuring iodine VPN connections
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * Copyright © 2012 Guido Günther <agx@sigxcpu.org>
 */

#ifndef _NM_IODINE_H_
#define _NM_IODINE_H_

#include <glib-object.h>

#define IODINE_TYPE_EDITOR_PLUGIN            (iodine_editor_plugin_get_type ())
#define IODINE_EDITOR_PLUGIN(obj)            (G_TYPE_CHECK_INSTANCE_CAST ((obj), IODINE_TYPE_EDITOR_PLUGIN, IodineEditorPlugin))
#define IODINE_EDITOR_PLUGIN_CLASS(klass)    (G_TYPE_CHECK_CLASS_CAST ((klass), IODINE_TYPE_EDITOR_PLUGIN, IodineEditorPluginClass))
#define IODINE_IS_EDITOR_PLUGIN(obj)         (G_TYPE_CHECK_INSTANCE_TYPE ((obj), IODINE_TYPE_EDITOR_PLUGIN))
#define IODINE_IS_EDITOR_PLUGIN_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE ((klass), IODINE_TYPE_EDITOR_PLUGIN))
#define IODINE_EDITOR_PLUGIN_GET_CLASS(obj)  (G_TYPE_INSTANCE_GET_CLASS ((obj), IODINE_TYPE_EDITOR_PLUGIN, IodineEditorPluginClass))

typedef struct _IodineEditorPlugin IodineEditorPlugin;
typedef struct _IodineEditorPluginClass IodineEditorPluginClass;

struct _IodineEditorPlugin {
	GObject parent;
};

struct _IodineEditorPluginClass {
	GObjectClass parent;
};

GType iodine_editor_plugin_get_type (void);


#define IODINE_TYPE_EDITOR            (iodine_editor_get_type ())
#define IODINE_EDITOR(obj)            (G_TYPE_CHECK_INSTANCE_CAST ((obj), IODINE_TYPE_EDITOR, IodineEditor))
#define IODINE_EDITOR_CLASS(klass)    (G_TYPE_CHECK_CLASS_CAST ((klass), IODINE_TYPE_EDITOR, IodineEditorClass))
#define IODINE_IS_EDITOR(obj)         (G_TYPE_CHECK_INSTANCE_TYPE ((obj), IODINE_TYPE_EDITOR))
#define IODINE_IS_EDITOR_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE ((klass), IODINE_TYPE_EDITOR))
#define IODINE_EDITOR_GET_CLASS(obj)  (G_TYPE_INSTANCE_GET_CLASS ((obj), IODINE_TYPE_EDITOR, IodineEditorClass))

typedef struct _IodineEditor IodineEditor;
typedef struct _IodineEditorClass IodineEditorClass;

struct _IodineEditor {
	GObject parent;
};

struct _IodineEditorClass {
	GObjectClass parent;
};

GType iodine_editor_get_type (void);

#endif	/* _NM_IODINE_H_ */

