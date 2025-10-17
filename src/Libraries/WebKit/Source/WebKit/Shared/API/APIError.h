/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 11, 2024.
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
#include <WebCore/ResourceError.h>

namespace API {

class Error : public ObjectImpl<Object::Type::Error> {
public:
    static Ref<Error> create()
    {
        return adoptRef(*new Error);
    }

    static Ref<Error> create(const WebCore::ResourceError& error)
    {
        return adoptRef(*new Error(error));
    }

    enum General {
        Internal = 300
    };
    static const WTF::String& webKitErrorDomain();

    enum Network {
        Cancelled = 302,
        FileDoesNotExist = 303,
        HTTPSUpgradeRedirectLoop = 304,
        HTTPNavigationWithHTTPSOnlyError = 305,
    };
    static const WTF::String& webKitNetworkErrorDomain();

    enum Policy {
        CannotShowMIMEType = 100,
        CannotShowURL = 101,
        FrameLoadInterruptedByPolicyChange = 102,
        CannotUseRestrictedPort = 103,
        FrameLoadBlockedByContentBlocker = 104,
        FrameLoadBlockedByContentFilter = 105,
        FrameLoadBlockedByRestrictions = 106,
    };
    static const WTF::String& webKitPolicyErrorDomain();

    enum Plugin {
        CannotFindPlugIn = 200,
        CannotLoadPlugIn = 201,
        JavaUnavailable = 202,
        PlugInCancelledConnection = 203,
        PlugInWillHandleLoad = 204,
        InsecurePlugInVersion = 205
    };
    static const WTF::String& webKitPluginErrorDomain();

#if USE(SOUP)
    enum Download {
        Transport = 499,
        CancelledByUser = 400,
        Destination = 401
    };
    static const WTF::String& webKitDownloadErrorDomain();
#endif

#if PLATFORM(GTK)
    enum Print {
        Generic = 599,
        PrinterNotFound = 500,
        InvalidPageRange = 501
    };
    static const WTF::String& webKitPrintErrorDomain();
#endif

    const WTF::String& domain() const { return m_platformError.domain(); }
    int errorCode() const { return m_platformError.errorCode(); }
    const WTF::String& failingURL() const { return m_platformError.failingURL().string(); }
    const WTF::String& localizedDescription() const { return m_platformError.localizedDescription(); }

    const WebCore::ResourceError& platformError() const { return m_platformError; }

private:
    Error()
    {
    }

    Error(const WebCore::ResourceError& error)
        : m_platformError(error)
    {
    }

    WebCore::ResourceError m_platformError;
};

} // namespace API

SPECIALIZE_TYPE_TRAITS_API_OBJECT(Error);
