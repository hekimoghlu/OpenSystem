/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 26, 2021.
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
#include "WKContextConfigurationRef.h"

#include "APIArray.h"
#include "APIProcessPoolConfiguration.h"
#include "OverrideLanguages.h"
#include "WKAPICast.h"

using namespace WebKit;

WKContextConfigurationRef WKContextConfigurationCreate()
{
    return toAPI(&API::ProcessPoolConfiguration::create().leakRef());
}

WKContextConfigurationRef WKContextConfigurationCreateWithLegacyOptions()
{
    return WKContextConfigurationCreate();
}

WKStringRef WKContextConfigurationCopyDiskCacheDirectory(WKContextConfigurationRef)
{
    return nullptr;
}

void WKContextConfigurationSetDiskCacheDirectory(WKContextConfigurationRef, WKStringRef)
{
}

WKStringRef WKContextConfigurationCopyIndexedDBDatabaseDirectory(WKContextConfigurationRef)
{
    return nullptr;
}

void WKContextConfigurationSetIndexedDBDatabaseDirectory(WKContextConfigurationRef, WKStringRef)
{
}

WKStringRef WKContextConfigurationCopyInjectedBundlePath(WKContextConfigurationRef configuration)
{
    return toCopiedAPI(toImpl(configuration)->injectedBundlePath());
}

void WKContextConfigurationSetInjectedBundlePath(WKContextConfigurationRef configuration, WKStringRef injectedBundlePath)
{
    toImpl(configuration)->setInjectedBundlePath(toImpl(injectedBundlePath)->string());
}

WKArrayRef WKContextConfigurationCopyCustomClassesForParameterCoder(WKContextConfigurationRef configuration)
{
    return toAPI(&API::Array::createStringArray(Vector<String>()).leakRef());
}

void WKContextConfigurationSetCustomClassesForParameterCoder(WKContextConfigurationRef configuration, WKArrayRef classesForCoder)
{
}

WKStringRef WKContextConfigurationCopyLocalStorageDirectory(WKContextConfigurationRef)
{
    return nullptr;
}

void WKContextConfigurationSetLocalStorageDirectory(WKContextConfigurationRef, WKStringRef)
{
}

WKStringRef WKContextConfigurationCopyWebSQLDatabaseDirectory(WKContextConfigurationRef)
{
    return nullptr;
}

void WKContextConfigurationSetWebSQLDatabaseDirectory(WKContextConfigurationRef, WKStringRef)
{
}

WKStringRef WKContextConfigurationCopyMediaKeysStorageDirectory(WKContextConfigurationRef)
{
    return nullptr;
}

void WKContextConfigurationSetMediaKeysStorageDirectory(WKContextConfigurationRef, WKStringRef)
{
}

WKStringRef WKContextConfigurationCopyResourceLoadStatisticsDirectory(WKContextConfigurationRef)
{
    return nullptr;
}

void WKContextConfigurationSetResourceLoadStatisticsDirectory(WKContextConfigurationRef, WKStringRef)
{
}

bool WKContextConfigurationFullySynchronousModeIsAllowedForTesting(WKContextConfigurationRef configuration)
{
    return toImpl(configuration)->fullySynchronousModeIsAllowedForTesting();
}

void WKContextConfigurationSetFullySynchronousModeIsAllowedForTesting(WKContextConfigurationRef configuration, bool allowed)
{
    toImpl(configuration)->setFullySynchronousModeIsAllowedForTesting(allowed);
}

bool WKContextConfigurationIgnoreSynchronousMessagingTimeoutsForTesting(WKContextConfigurationRef configuration)
{
    return toImpl(configuration)->ignoreSynchronousMessagingTimeoutsForTesting();
}

void WKContextConfigurationSetIgnoreSynchronousMessagingTimeoutsForTesting(WKContextConfigurationRef configuration, bool ignore)
{
    toImpl(configuration)->setIgnoreSynchronousMessagingTimeoutsForTesting(ignore);
}

WKArrayRef WKContextConfigurationCopyOverrideLanguages(WKContextConfigurationRef)
{
    // FIXME: Delete this function.
    return toAPI(&API::Array::create().leakRef());
}

void WKContextConfigurationSetOverrideLanguages(WKContextConfigurationRef, WKArrayRef overrideLanguages)
{
    // FIXME: This is an SPI function, and is only (supposed to be) used for testing.
    // However, playwright automation tests rely on it.
    // See https://bugs.webkit.org/show_bug.cgi?id=242827 for details.
    WebKit::setOverrideLanguages(toImpl(overrideLanguages)->toStringVector());
}

bool WKContextConfigurationProcessSwapsOnNavigation(WKContextConfigurationRef configuration)
{
    return toImpl(configuration)->processSwapsOnNavigation();
}

void WKContextConfigurationSetProcessSwapsOnNavigation(WKContextConfigurationRef configuration, bool swaps)
{
    toImpl(configuration)->setProcessSwapsOnNavigation(swaps);
}

bool WKContextConfigurationPrewarmsProcessesAutomatically(WKContextConfigurationRef configuration)
{
    return toImpl(configuration)->isAutomaticProcessWarmingEnabled();
}

void WKContextConfigurationSetPrewarmsProcessesAutomatically(WKContextConfigurationRef configuration, bool prewarms)
{
    toImpl(configuration)->setIsAutomaticProcessWarmingEnabled(prewarms);
}

bool WKContextConfigurationUsesWebProcessCache(WKContextConfigurationRef configuration)
{
    return toImpl(configuration)->usesWebProcessCache();
}

void WKContextConfigurationSetUsesWebProcessCache(WKContextConfigurationRef configuration, bool uses)
{
    toImpl(configuration)->setUsesWebProcessCache(uses);
}

bool WKContextConfigurationAlwaysKeepAndReuseSwappedProcesses(WKContextConfigurationRef configuration)
{
    return toImpl(configuration)->alwaysKeepAndReuseSwappedProcesses();
}

void WKContextConfigurationSetAlwaysKeepAndReuseSwappedProcesses(WKContextConfigurationRef configuration, bool keepAndReuse)
{
    toImpl(configuration)->setAlwaysKeepAndReuseSwappedProcesses(keepAndReuse);
}

int64_t WKContextConfigurationDiskCacheSizeOverride(WKContextConfigurationRef configuration)
{
    return 0;
}

void WKContextConfigurationSetDiskCacheSizeOverride(WKContextConfigurationRef configuration, int64_t size)
{
}

void WKContextConfigurationSetShouldCaptureAudioInUIProcess(WKContextConfigurationRef, bool)
{
}

void WKContextConfigurationSetShouldConfigureJSCForTesting(WKContextConfigurationRef configuration, bool value)
{
    toImpl(configuration)->setShouldConfigureJSCForTesting(value);
}

WKStringRef WKContextConfigurationCopyTimeZoneOverride(WKContextConfigurationRef configuration)
{
    return toCopiedAPI(toImpl(configuration)->timeZoneOverride());
}

void WKContextConfigurationSetTimeZoneOverride(WKContextConfigurationRef configuration, WKStringRef timeZoneOverride)
{
    toImpl(configuration)->setTimeZoneOverride(toImpl(timeZoneOverride)->string());
}
