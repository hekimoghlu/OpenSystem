/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 20, 2025.
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
/*
 * EAPWiFiUtil.m
 * - C wrapper functions over ObjC interface to CoreWiFi Framework
 */

#import <Foundation/Foundation.h>
#import <CoreWiFi/CoreWiFi.h>
#if TARGET_OS_IPHONE
#include "EAPOLSIMPrefsManage.h"
#endif /* TARGET_OS_IPHONE */
#include "EAPLog.h"
#include "EAPWiFiUtil.h"

static Boolean S_wifi_power_state;

static void
EAPWiFiHandlePowerStatusChange(Boolean powered_on)
{
    /*
     * increment the generation ID in SC prefs so eapclient would know
     * that wifi power was toggled from ON to OFF and it should not
     * use the SIM specific stored info.
     * So turning WiFi power off is similar to ejecting SIM as both actions
     * lead to tearing down the 802.1X connection and incrementing the
     * generation ID.
     */
    if (S_wifi_power_state == TRUE && powered_on == FALSE) {
	EAPLOG_FL(LOG_INFO, "WiFi power is turned off");
	EAPOLSIMGenerationIncrement();
    }
    S_wifi_power_state = powered_on;
}

static void
EAPWiFiHandleCWFEvents(CWFInterface *cwfInterface, CWFEvent *event)
{
    @autoreleasepool {
	switch (event.type) {
	    case CWFEventTypePowerChanged:
	    {
		Boolean on = cwfInterface.powerOn ? TRUE : FALSE;
		EAPLOG_FL(LOG_DEBUG, "power state changed to %s", on ? "ON" : "OFF");
		EAPWiFiHandlePowerStatusChange(on);
	    }
	    default:
		break;
	}
    }
    return;
}

void
EAPWiFiMonitorPowerStatus(void)
{
#if TARGET_OS_IOS || TARGET_OS_WATCH
    @autoreleasepool {
	dispatch_queue_t 	queue = dispatch_queue_create("EAP WiFi Interface Queue", NULL);
	CWFInterface 		*cwfInterface = [[CWFInterface alloc] init];

	cwfInterface.eventHandler = ^( CWFEvent *cwfEvent ) {
	    dispatch_async(queue, ^{
		EAPWiFiHandleCWFEvents(cwfInterface, cwfEvent);
	    });
	};
	dispatch_async(queue, ^{
	    [cwfInterface activate];
	    S_wifi_power_state = cwfInterface.powerOn ? TRUE : FALSE;
	    [cwfInterface startMonitoringEventType:CWFEventTypePowerChanged error:nil];
	});
    }
#endif
    return;
}
