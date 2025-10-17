/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 16, 2025.
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

#include <QSize>
#include <QUrl>
#include <wpe/webkit.h>
#include <wtf/glib/GRefPtr.h>

class WPEQtViewPrivate {
public:
    WPEQtViewPrivate() { }
    ~WPEQtViewPrivate() { }

    GRefPtr<WebKitWebView> m_webView;
    QUrl m_url;
    QString m_html;
    QUrl m_baseUrl;
    QSize m_size { 800, 600 };
    bool m_errorOccured { false };
};

Q_DECLARE_METATYPE(WPEQtViewPrivate)
