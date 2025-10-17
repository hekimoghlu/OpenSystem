/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 19, 2023.
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

#include "APIObject.h"
#include <WebCore/FileChooser.h>

namespace API {

class Array;

class OpenPanelParameters : public API::ObjectImpl<API::Object::Type::OpenPanelParameters> {
public:
    static Ref<OpenPanelParameters> create(const WebCore::FileChooserSettings&);
    ~OpenPanelParameters();

    bool allowDirectories() const { return m_settings.allowsDirectories; }
    bool allowMultipleFiles() const { return m_settings.allowsMultipleFiles; }
    Ref<API::Array> acceptMIMETypes() const;
    Ref<API::Array> acceptFileExtensions() const;
    Ref<API::Array> allowedMIMETypes() const;
    Ref<API::Array> allowedFileExtensions() const;
    Ref<API::Array> selectedFileNames() const;
#if ENABLE(MEDIA_CAPTURE)
    WebCore::MediaCaptureType mediaCaptureType() const { return m_settings.mediaCaptureType; }
#endif

private:
    explicit OpenPanelParameters(const WebCore::FileChooserSettings&);

    WebCore::FileChooserSettings m_settings;
};

} // namespace API
