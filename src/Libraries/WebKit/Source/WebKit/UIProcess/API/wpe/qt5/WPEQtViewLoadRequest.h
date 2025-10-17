/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 4, 2025.
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

#include "WPEQtView.h"

#include <QObject>

class WPEQtViewLoadRequestPrivate;

class Q_DECL_EXPORT WPEQtViewLoadRequest : public QObject {
    Q_OBJECT
    Q_PROPERTY(QUrl url READ url)
    Q_PROPERTY(WPEQtView::LoadStatus status READ status)
    Q_PROPERTY(QString errorString READ errorString)

public:
    ~WPEQtViewLoadRequest();

    QUrl url() const;
    WPEQtView::LoadStatus status() const;
    QString errorString() const;

    explicit WPEQtViewLoadRequest(const WPEQtViewLoadRequestPrivate&);

private:
    friend class WPEQtView;

    Q_DECLARE_PRIVATE(WPEQtViewLoadRequest)
    QScopedPointer<WPEQtViewLoadRequestPrivate> d_ptr;
};

