/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 9, 2024.
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
 * IPConfigurationUtil.c
 * - API to communicate with IPConfiguration agent to perform various tasks
 */

/* 
 * Modification History
 *
 * March 29, 2018 	Dieter Siegmund (dieter@apple.com)
 * - initial revision
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <mach/mach.h>
#include <mach/mach_error.h>
#include <CoreFoundation/CoreFoundation.h>
#include "IPConfigurationUtil.h"
#include "IPConfigurationUtilPrivate.h"
#include "ipconfig_types.h"
#include "ipconfig_ext.h"
#include "ipconfig.h"
#include "cfutil.h"
#include "IPConfigurationLog.h"
#include "symbol_scope.h"
#include "IPConfigurationPrivate.h"

STATIC CFDictionaryRef
create_network_dict(CFStringRef ssid)
{
    const void *	key = kIPConfigurationForgetNetworkSSID;
    const void *	value = ssid;

    return (CFDictionaryCreate(NULL, &key, &value, 1,
			       &kCFTypeDictionaryKeyCallBacks,
			       &kCFTypeDictionaryValueCallBacks));
}

Boolean
IPConfigurationForgetNetwork(CFStringRef interface_name, CFStringRef ssid)
{
    InterfaceName		ifname;
    kern_return_t		kret;
    CFDictionaryRef		network_dict;
    CFDataRef			network_data;
    mach_port_t			server = MACH_PORT_NULL;
    ipconfig_status_t		status;
    Boolean			success = FALSE;
    void *			xml_data_ptr = NULL;
    int				xml_data_len = 0;

    _IPConfigurationInitLog(kIPConfigurationLogCategoryLibrary);

    if (interface_name == NULL || ssid == NULL) {
	IPConfigLog(LOG_NOTICE, "%s: interface and SSID must not be NULL",
		    __func__);
	return (FALSE);
    }

    kret = ipconfig_server_port(&server);
    if (kret != BOOTSTRAP_SUCCESS) {
	IPConfigLog(LOG_NOTICE,
		    "ipconfig_server_port, %s",
		    mach_error_string(kret));
	return (FALSE);
    }
    InterfaceNameInitWithCFString(ifname, interface_name);
    network_dict = create_network_dict(ssid);
    network_data
	= CFPropertyListCreateData(NULL,
				   network_dict,
				   kCFPropertyListBinaryFormat_v1_0,
				   0,
				   NULL);
    CFRelease(network_dict);
    xml_data_ptr = (void *)CFDataGetBytePtr(network_data);
    xml_data_len = (int)CFDataGetLength(network_data);
    kret = ipconfig_forget_network(server, ifname,
				   xml_data_ptr, xml_data_len,
				   &status);
    CFRelease(network_data);
    if (kret != KERN_SUCCESS) {
	IPConfigLog(LOG_NOTICE,
		    "ipconfig_forget_network(%s) failed, %s",
		    ifname, mach_error_string(kret));
    }
    else if (status != ipconfig_status_success_e) {
	IPConfigLog(LOG_NOTICE,
		    "ipconfig_forget_network(%s) failed, %s",
		    ifname, ipconfig_status_string(status));
    }
    else {
	IPConfigLog(LOG_NOTICE,
		    "ipconfig_forget_network(%s) succeeded", ifname);
	success = TRUE;
    }
    return (success);
}
