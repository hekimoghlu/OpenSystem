/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 2, 2022.
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

#include <utility>
#include <wtf/Forward.h>
#include <wtf/URL.h>
#include <wtf/Vector.h>
#include <wtf/text/WTFString.h>

namespace WebDriver {

struct Timeouts {
    std::optional<double> script;
    std::optional<double> pageLoad;
    std::optional<double> implicit;
};

enum class PageLoadStrategy {
    None,
    Normal,
    Eager
};

enum class UnhandledPromptBehavior {
    Dismiss,
    Accept,
    DismissAndNotify,
    AcceptAndNotify,
    Ignore
};

struct Proxy {
    String type;
    std::optional<URL> autoconfigURL;
    std::optional<URL> ftpURL;
    std::optional<URL> httpURL;
    std::optional<URL> httpsURL;
    std::optional<URL> socksURL;
    std::optional<uint8_t> socksVersion;
    Vector<String> ignoreAddressList;
};

struct Capabilities {
    std::optional<String> browserName;
    std::optional<String> browserVersion;
    std::optional<String> platformName;
    std::optional<bool> acceptInsecureCerts;
    std::optional<bool> strictFileInteractability;
    std::optional<bool> setWindowRect;
    std::optional<Timeouts> timeouts;
    std::optional<PageLoadStrategy> pageLoadStrategy;
    std::optional<UnhandledPromptBehavior> unhandledPromptBehavior;
    std::optional<Proxy> proxy;
    std::optional<String> targetAddr;
    std::optional<int> targetPort;
#if PLATFORM(GTK) || PLATFORM(WPE)
    std::optional<String> browserBinary;
    std::optional<Vector<String>> browserArguments;
    std::optional<Vector<std::pair<String, String>>> certificates;
#endif
#if PLATFORM(GTK)
    std::optional<bool> useOverlayScrollbars;
#endif
    // https://w3c.github.io/webdriver-bidi/#websocket-url
    std::optional<bool> webSocketURL;
};

} // namespace WebDriver
