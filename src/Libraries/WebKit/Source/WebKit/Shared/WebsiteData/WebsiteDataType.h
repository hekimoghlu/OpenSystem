/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 6, 2024.
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

namespace WebKit {

enum class WebsiteDataType : uint32_t {
    Cookies = 1 << 0,
    DiskCache = 1 << 1,
    MemoryCache = 1 << 2,
    OfflineWebApplicationCache = 1 << 3,
    SessionStorage = 1 << 4,
    LocalStorage = 1 << 5,
    WebSQLDatabases = 1 << 6,
    IndexedDBDatabases = 1 << 7,
    MediaKeys = 1 << 8,
    HSTSCache = 1 << 9,
    SearchFieldRecentSearches = 1 << 10,
    ResourceLoadStatistics = 1 << 12,
    Credentials = 1 << 13,
    ServiceWorkerRegistrations = 1 << 14,
    DOMCache = 1 << 15,
    DeviceIdHashSalt = 1 << 16,
    PrivateClickMeasurements = 1 << 17,
#if HAVE(ALTERNATIVE_SERVICE)
    AlternativeServices = 1 << 18,
#endif
    FileSystem = 1 << 19,
    BackgroundFetchStorage = 1 << 20,
};

} // namespace WebKit

namespace WTF {

template<> struct EnumTraitsForPersistence<WebKit::WebsiteDataType> {
    using values = EnumValues<
        WebKit::WebsiteDataType,
        WebKit::WebsiteDataType::Cookies,
        WebKit::WebsiteDataType::DiskCache,
        WebKit::WebsiteDataType::MemoryCache,
        WebKit::WebsiteDataType::OfflineWebApplicationCache,
        WebKit::WebsiteDataType::SessionStorage,
        WebKit::WebsiteDataType::LocalStorage,
        WebKit::WebsiteDataType::WebSQLDatabases,
        WebKit::WebsiteDataType::IndexedDBDatabases,
        WebKit::WebsiteDataType::MediaKeys,
        WebKit::WebsiteDataType::HSTSCache,
        WebKit::WebsiteDataType::SearchFieldRecentSearches,
        WebKit::WebsiteDataType::ResourceLoadStatistics,
        WebKit::WebsiteDataType::Credentials,
        WebKit::WebsiteDataType::ServiceWorkerRegistrations,
        WebKit::WebsiteDataType::DOMCache,
        WebKit::WebsiteDataType::DeviceIdHashSalt,
        WebKit::WebsiteDataType::PrivateClickMeasurements,
#if HAVE(ALTERNATIVE_SERVICE)
        WebKit::WebsiteDataType::AlternativeServices,
#endif
        WebKit::WebsiteDataType::FileSystem,
        WebKit::WebsiteDataType::BackgroundFetchStorage
    >;
};

} // namespace WTF
