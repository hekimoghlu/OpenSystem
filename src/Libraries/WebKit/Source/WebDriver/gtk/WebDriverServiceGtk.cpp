/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 30, 2025.
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
#include "WebDriverService.h"

#include "Capabilities.h"
#include "CommandResult.h"
#include <wtf/JSONValues.h>
#include <wtf/text/StringToIntegerConversion.h>

namespace WebDriver {

void WebDriverService::platformInit()
{
}

Capabilities WebDriverService::platformCapabilities()
{
    Capabilities capabilities;
    capabilities.platformName = String("linux"_s);
    capabilities.setWindowRect = true;
    return capabilities;
}

bool WebDriverService::platformValidateCapability(const String& name, const Ref<JSON::Value>& value) const
{
    if (name != "webkitgtk:browserOptions"_s)
        return true;

    auto browserOptions = value->asObject();
    if (!browserOptions)
        return false;

    if (browserOptions->isNull())
        return true;

    if (auto binaryValue = browserOptions->getValue("binary"_s)) {
        if (!binaryValue->asString())
            return false;
    }

    if (auto useOverlayScrollbarsValue = browserOptions->getValue("useOverlayScrollbars"_s)) {
        if (!useOverlayScrollbarsValue->asBoolean())
            return false;
    }

    if (auto browserArgumentsValue = browserOptions->getValue("args"_s)) {
        auto browserArguments = browserArgumentsValue->asArray();
        if (!browserArguments)
            return false;

        unsigned browserArgumentsLength = browserArguments->length();
        for (unsigned i = 0; i < browserArgumentsLength; ++i) {
            auto argument = browserArguments->get(i)->asString();
            if (!argument)
                return false;
        }
    }

    if (auto certificatesValue = browserOptions->getValue("certificates"_s)) {
        auto certificates = certificatesValue->asArray();
        if (!certificates)
            return false;

        unsigned certificatesLength = certificates->length();
        for (unsigned i = 0; i < certificatesLength; ++i) {
            auto certificate = certificates->get(i)->asObject();
            if (!certificate)
                return false;

            auto host = certificate->getString("host"_s);
            if (!host)
                return false;

            auto certificateFile = certificate->getString("certificateFile"_s);
            if (!certificateFile)
                return false;
        }
    }

    return true;
}

bool WebDriverService::platformMatchCapability(const String&, const Ref<JSON::Value>&) const
{
    return true;
}

void WebDriverService::platformParseCapabilities(const JSON::Object& matchedCapabilities, Capabilities& capabilities) const
{
    capabilities.browserBinary = String(LIBEXECDIR "/webkit" WEBKITGTK_API_INFIX "gtk-" WEBKITGTK_API_VERSION "/MiniBrowser"_s);
    capabilities.browserArguments = Vector<String> { "--automation"_s };
    capabilities.useOverlayScrollbars = true;

    auto browserOptions = matchedCapabilities.getObject("webkitgtk:browserOptions"_s);
    if (!browserOptions)
        return;

    auto browserBinary = browserOptions->getString("binary"_s);
    if (!!browserBinary) {
        capabilities.browserBinary = browserBinary;
        capabilities.browserArguments = std::nullopt;
    }

    auto browserArguments = browserOptions->getArray("args"_s);
    if (browserArguments && browserArguments->length()) {
        unsigned browserArgumentsLength = browserArguments->length();
        capabilities.browserArguments = Vector<String>();
        capabilities.browserArguments->reserveInitialCapacity(browserArgumentsLength);
        for (unsigned i = 0; i < browserArgumentsLength; ++i) {
            auto argument = browserArguments->get(i)->asString();
            ASSERT(!argument.isNull());
            capabilities.browserArguments->append(WTFMove(argument));
        }
    }

    if (auto useOverlayScrollbars = browserOptions->getBoolean("useOverlayScrollbars"_s))
        capabilities.useOverlayScrollbars = useOverlayScrollbars;
    else
        capabilities.useOverlayScrollbars = true;

    auto certificates = browserOptions->getArray("certificates"_s);
    if (certificates && certificates->length()) {
        unsigned certificatesLength = certificates->length();
        capabilities.certificates = Vector<std::pair<String, String>>();
        capabilities.certificates->reserveInitialCapacity(certificatesLength);
        for (unsigned i = 0; i < certificatesLength; ++i) {
            auto certificate = certificates->get(i)->asObject();
            ASSERT(certificate);

            auto host = certificate->getString("host"_s);
            ASSERT(!host.isNull());

            auto certificateFile = certificate->getString("certificateFile"_s);
            ASSERT(!certificateFile.isNull());

            capabilities.certificates->append({ WTFMove(host), WTFMove(certificateFile) });
        }
    }

    String targetString;
    if (browserOptions->getString("targetAddress"_s, targetString)) {
        auto position = targetString.reverseFind(':');
        if (position != notFound) {
            capabilities.targetAddr = targetString.left(position);
            capabilities.targetPort = parseIntegerAllowingTrailingJunk<uint16_t>(StringView { targetString }.substring(position + 1)).value_or(0);
        }
    }
}

} // namespace WebDriver
