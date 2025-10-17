/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 28, 2022.
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

#include <WebCore/PrivateClickMeasurement.h>

namespace WebKit {

enum class PrivateClickMeasurementAttributionType : bool;

namespace PCM {

class Database;
struct DebugInfo;

class Store : public ThreadSafeRefCounted<Store> {
public:
    virtual ~Store() = default;

    using ApplicationBundleIdentifier = String;

    virtual void insertPrivateClickMeasurement(WebCore::PrivateClickMeasurement&&, WebKit::PrivateClickMeasurementAttributionType, CompletionHandler<void()>&&) = 0;
    virtual void attributePrivateClickMeasurement(WebCore::PCM::SourceSite&&, WebCore::PCM::AttributionDestinationSite&&, const ApplicationBundleIdentifier&, WebCore::PCM::AttributionTriggerData&&, WebCore::PrivateClickMeasurement::IsRunningLayoutTest, CompletionHandler<void(std::optional<WebCore::PCM::AttributionSecondsUntilSendData>&&, DebugInfo&&)>&&) = 0;

    virtual void privateClickMeasurementToStringForTesting(CompletionHandler<void(String)>&&) const = 0;
    virtual void markAllUnattributedPrivateClickMeasurementAsExpiredForTesting() = 0;
    virtual void markAttributedPrivateClickMeasurementsAsExpiredForTesting(CompletionHandler<void()>&&) = 0;

    virtual void allAttributedPrivateClickMeasurement(CompletionHandler<void(Vector<WebCore::PrivateClickMeasurement>&&)>&&) = 0;
    virtual void clearExpiredPrivateClickMeasurement() = 0;
    virtual void clearPrivateClickMeasurement(CompletionHandler<void()>&&) = 0;
    virtual void clearPrivateClickMeasurementForRegistrableDomain(WebCore::RegistrableDomain&&, CompletionHandler<void()>&&) = 0;
    virtual void clearSentAttribution(WebCore::PrivateClickMeasurement&& attributionToClear, WebCore::PCM::AttributionReportEndpoint) = 0;

    virtual void close(CompletionHandler<void()>&&) = 0;
};

} // namespace PCM

} // namespace WebKit
