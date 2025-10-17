/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 12, 2024.
 *
 * Licensed under the Apache License, Version 2.0 (the ""License"");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an ""AS IS"" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please contact NeXTHub Corporation, 651 N Broad St, Suite 201, 
 * Middletown, DE 19709, New Castle County, USA.
 *
 */
#ifndef WTF_GTypedefs_h
#define WTF_GTypedefs_h

/* Vanilla C code does not seem to be able to handle forward-declaration typedefs. */
#ifdef  __cplusplus

typedef char gchar;
typedef double gdouble;
typedef float gfloat;
typedef int gint;
typedef gint gboolean;
typedef long glong;
typedef short gshort;
typedef unsigned char guchar;
typedef unsigned int guint;
typedef unsigned long gulong;
typedef unsigned short gushort;
typedef void* gpointer;

typedef struct _GAsyncResult GAsyncResult;
typedef struct _GCancellable GCancellable;
typedef struct _GCharsetConverter GCharsetConverter;
typedef struct _GDir GDir;
typedef struct _GdkAtom* GdkAtom;
typedef struct _GdkCursor GdkCursor;
typedef struct _GdkDragContext GdkDragContext;
typedef struct _GdkEventConfigure GdkEventConfigure;
typedef struct _GdkEventExpose GdkEventExpose;
typedef struct _GdkPixbuf GdkPixbuf;
typedef struct _GError GError;
typedef struct _GFile GFile;
typedef struct _GHashTable GHashTable;
typedef struct _GInputStream GInputStream;
typedef struct _GList GList;
typedef struct _GMainContext GMainContext;
typedef struct _GMainLoop GMainLoop;
typedef struct _GPatternSpec GPatternSpec;
typedef struct _GPollableOutputStream GPollableOutputStream;
typedef struct _GResource GResource;
typedef struct _GSList GSList;
typedef struct _GSocketClient GSocketClient;
typedef struct _GSocketConnection GSocketConnection;
typedef struct _GSource GSource;
typedef struct _GVariant GVariant;
typedef struct _GVariantBuilder GVariantBuilder;
typedef struct _GVariantIter GVariantIter;
typedef struct _GVariantType GVariantType;
#if USE(GTK4)
typedef struct _GdkEvent GdkEvent;
#else
typedef union _GdkEvent GdkEvent;
#endif
typedef struct _GTimer GTimer;
typedef struct _GKeyFile GKeyFile;
typedef struct _GPtrArray GPtrArray;
typedef struct _GByteArray GByteArray;
typedef struct _GBytes GBytes;
typedef struct _GClosure GClosure;

#if USE(CAIRO)
typedef struct _cairo_surface cairo_surface_t;
typedef struct _cairo_rectangle_int cairo_rectangle_int_t;
#endif

#if PLATFORM(GTK)
typedef struct _GtkAction GtkAction;
typedef struct _GtkAdjustment GtkAdjustment;
typedef struct _GtkBorder GtkBorder;
typedef struct _GtkClipboard GtkClipboard;
typedef struct _GtkContainer GtkContainer;
typedef struct _GtkMenu GtkMenu;
typedef struct _GtkMenuItem GtkMenuItem;
typedef struct _GtkObject GtkObject;
typedef struct _GtkSelectionData GtkSelectionData;
typedef struct _GtkStyle GtkStyle;
typedef struct _GtkTargetList GtkTargetList;
typedef struct _GtkThemeParts GtkThemeParts;
typedef struct _GtkWidget GtkWidget;
typedef struct _GtkWindow GtkWindow;
typedef struct _GdkWindow GdkWindow;
typedef struct _GtkStyleContext GtkStyleContext;
#endif

#endif
#endif // WTF_GTypedefs_h
