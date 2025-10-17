/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 13, 2021.
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

#if ENABLE(ENCRYPTED_MEDIA) && ENABLE(THUNDER) && USE(GSTREAMER)

#include "WebKitCommonEncryptionDecryptorGStreamer.h"

G_BEGIN_DECLS

#define WEBKIT_TYPE_MEDIA_THUNDER_DECRYPT          (webkit_media_thunder_decrypt_get_type())
#define WEBKIT_MEDIA_THUNDER_DECRYPT(obj)          (G_TYPE_CHECK_INSTANCE_CAST((obj), WEBKIT_TYPE_MEDIA_THUNDER_DECRYPT, WebKitMediaThunderDecrypt))
#define WEBKIT_MEDIA_THUNDER_DECRYPT_CLASS(klass)  (G_TYPE_CHECK_CLASS_CAST((klass), WEBKIT_TYPE_MEDIA_THUNDER_DECRYPT, WebKitMediaThunderDecryptClass))
#define WEBKIT_IS_MEDIA_THUNDER_DECRYPT(obj)       (G_TYPE_CHECK_INSTANCE_TYPE((obj), WEBKIT_TYPE_MEDIA_THUNDER_DECRYPT))
#define WEBKIT_IS_MEDIA_THUNDER_DECRYPT_CLASS(obj) (G_TYPE_CHECK_CLASS_TYPE((klass), WEBKIT_TYPE_MEDIA_THUNDER_DECRYPT))

typedef struct _WebKitMediaThunderDecrypt        WebKitMediaThunderDecrypt;
typedef struct _WebKitMediaThunderDecryptClass   WebKitMediaThunderDecryptClass;
struct WebKitMediaThunderDecryptPrivate;

GType webkit_media_thunder_decrypt_get_type(void);

struct _WebKitMediaThunderDecrypt {
    WebKitMediaCommonEncryptionDecrypt parent;

    WebKitMediaThunderDecryptPrivate* priv;
};

struct _WebKitMediaThunderDecryptClass {
    WebKitMediaCommonEncryptionDecryptClass parentClass;
};

G_END_DECLS

#endif // ENABLE(ENCRYPTED_MEDIA) && ENABLE(THUNDER) && USE(GSTREAMER)
