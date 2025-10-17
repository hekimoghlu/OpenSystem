/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 1, 2023.
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
#include "config.h"
#include "APIOpenPanelParameters.h"

#include "APIArray.h"
#include "APIString.h"
#include <WebCore/MIMETypeRegistry.h>
#include <wtf/Vector.h>

namespace API {
using namespace WebCore;

Ref<OpenPanelParameters> OpenPanelParameters::create(const FileChooserSettings& settings)
{
    return adoptRef(*new OpenPanelParameters(settings));
}

OpenPanelParameters::OpenPanelParameters(const FileChooserSettings& settings)
    : m_settings(settings)
{
}

OpenPanelParameters::~OpenPanelParameters()
{
}

Ref<API::Array> OpenPanelParameters::acceptMIMETypes() const
{
    return API::Array::createStringArray(m_settings.acceptMIMETypes);
}

Ref<API::Array> OpenPanelParameters::acceptFileExtensions() const
{
    return API::Array::createStringArray(m_settings.acceptFileExtensions);
}

Ref<API::Array> OpenPanelParameters::allowedMIMETypes() const
{
    return API::Array::createStringArray(WebCore::MIMETypeRegistry::allowedMIMETypes(m_settings.acceptMIMETypes, m_settings.acceptFileExtensions));
}

Ref<API::Array> OpenPanelParameters::allowedFileExtensions() const
{
#if PLATFORM(MAC)
    auto acceptMIMETypes = m_settings.acceptMIMETypes;

    // On macOS allow selecting HEIF/HEIC images if acceptMIMETypes or acceptFileExtensions include at least
    // one MIME type which CG supports encoding to.
    if (MIMETypeRegistry::containsImageMIMETypeForEncoding(acceptMIMETypes, m_settings.acceptFileExtensions)) {
        acceptMIMETypes.append("image/heif"_s);
        acceptMIMETypes.append("image/heic"_s);
    }
    
    return API::Array::createStringArray(WebCore::MIMETypeRegistry::allowedFileExtensions(acceptMIMETypes, m_settings.acceptFileExtensions));
#else
    return API::Array::createStringArray(WebCore::MIMETypeRegistry::allowedFileExtensions(m_settings.acceptMIMETypes, m_settings.acceptFileExtensions));
#endif
}

Ref<API::Array> OpenPanelParameters::selectedFileNames() const
{
    return API::Array::createStringArray(m_settings.selectedFiles);
}

} // namespace WebCore
