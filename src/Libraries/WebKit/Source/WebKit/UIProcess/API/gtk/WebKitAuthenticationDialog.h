/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 21, 2022.
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
#pragma once

#include "WebKitAuthenticationRequest.h"
#include "WebKitWebViewDialog.h"
#include <gtk/gtk.h>

enum CredentialStorageMode {
    AllowPersistentStorage, // The user is asked whether to store credential information.
    DisallowPersistentStorage // Credential information is only kept in the session.
};

G_BEGIN_DECLS

#define WEBKIT_TYPE_AUTHENTICATION_DIALOG            (webkit_authentication_dialog_get_type())
#define WEBKIT_AUTHENTICATION_DIALOG(obj)            (G_TYPE_CHECK_INSTANCE_CAST((obj), WEBKIT_TYPE_AUTHENTICATION_DIALOG, WebKitAuthenticationDialog))
#define WEBKIT_IS_AUTHENTICATION_DIALOG(obj)         (G_TYPE_CHECK_INSTANCE_TYPE((obj), WEBKIT_TYPE_AUTHENTICATION_DIALOG))
#define WEBKIT_AUTHENTICATION_DIALOG_CLASS(klass)    (G_TYPE_CHECK_CLASS_CAST((klass),  WEBKIT_TYPE_AUTHENTICATION_DIALOG, WebKitAuthenticationDialogClass))
#define WEBKIT_IS_AUTHENTICATION_DIALOG_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE((klass),  WEBKIT_TYPE_AUTHENTICATION_DIALOG))
#define WEBKIT_AUTHENTICATION_DIALOG_GET_CLASS(obj)  (G_TYPE_INSTANCE_GET_CLASS((obj),  WEBKIT_TYPE_AUTHENTICATION_DIALOG, WebKitAuthenticationDialogClass))

typedef struct _WebKitAuthenticationDialog        WebKitAuthenticationDialog;
typedef struct _WebKitAuthenticationDialogClass   WebKitAuthenticationDialogClass;
typedef struct _WebKitAuthenticationDialogPrivate WebKitAuthenticationDialogPrivate;

struct _WebKitAuthenticationDialog {
    WebKitWebViewDialog parent;

    /*< private >*/
    WebKitAuthenticationDialogPrivate* priv;
};

struct _WebKitAuthenticationDialogClass {
    WebKitWebViewDialogClass parentClass;
};

GType webkit_authentication_dialog_get_type();
GtkWidget* webkitAuthenticationDialogNew(WebKitAuthenticationRequest*, CredentialStorageMode);

G_END_DECLS
