/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 6, 2024.
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
#include "WKOpenPanelParametersRef.h"

#include "APIArray.h"
#include "APIOpenPanelParameters.h"
#include "WKAPICast.h"

using namespace WebKit;

WKTypeID WKOpenPanelParametersGetTypeID()
{
    return toAPI(API::OpenPanelParameters::APIType);
}

bool WKOpenPanelParametersGetAllowsDirectories(WKOpenPanelParametersRef parametersRef)
{
    return toImpl(parametersRef)->allowDirectories();
}

bool WKOpenPanelParametersGetAllowsMultipleFiles(WKOpenPanelParametersRef parametersRef)
{
    return toImpl(parametersRef)->allowMultipleFiles();
}

WKArrayRef WKOpenPanelParametersCopyAcceptedMIMETypes(WKOpenPanelParametersRef parametersRef)
{
    return toAPI(&toImpl(parametersRef)->acceptMIMETypes().leakRef());
}

WKArrayRef WKOpenPanelParametersCopyAcceptedFileExtensions(WKOpenPanelParametersRef parametersRef)
{
    return toAPI(&toImpl(parametersRef)->acceptFileExtensions().leakRef());
}

WKArrayRef WKOpenPanelParametersCopyAllowedMIMETypes(WKOpenPanelParametersRef parametersRef)
{
    return toAPI(&toImpl(parametersRef)->allowedMIMETypes().leakRef());
}

// Deprecated.
WKStringRef WKOpenPanelParametersCopyCapture(WKOpenPanelParametersRef)
{
    return 0;
}

bool WKOpenPanelParametersGetMediaCaptureType(WKOpenPanelParametersRef parametersRef)
{
#if ENABLE(MEDIA_CAPTURE)
    return toImpl(parametersRef)->mediaCaptureType() != WebCore::MediaCaptureType::MediaCaptureTypeNone;
#else
    UNUSED_PARAM(parametersRef);
    return false;
#endif
}

WKArrayRef WKOpenPanelParametersCopySelectedFileNames(WKOpenPanelParametersRef parametersRef)
{
    return toAPI(&toImpl(parametersRef)->selectedFileNames().leakRef());
}
