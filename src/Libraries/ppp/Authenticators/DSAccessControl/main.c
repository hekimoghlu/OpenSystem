/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 5, 2024.
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
/* -----------------------------------------------------------------------------
 *
 *  Theory of operation :
 *
 *  plugin to add support for authentication through Directory Services.
 *
----------------------------------------------------------------------------- */


#include <stdio.h>

#include <syslog.h>
#include <sys/types.h>
#include <sys/time.h>
#include <string.h>
#include <membership.h>
#include <membershipPriv.h>
#include <CoreFoundation/CoreFoundation.h>

#include <DirectoryService/DirServices.h>
#include <DirectoryService/DirServicesUtils.h>
#include <DirectoryService/DirServicesConst.h>
#include <CoreFoundation/CFString.h>
#include <SystemConfiguration/SystemConfiguration.h>
#include "../../Helpers/pppd/pppd.h"
#include "../../Helpers/pppd/pppd.h"
#include "../DSAuthentication/DSUser.h"

#define VPN_SERVICE_NAME	"vpn"

static CFBundleRef 	bundle = 0;

static int dsaccess_authorize_user(u_char* user_name, int len);

/* -----------------------------------------------------------------------------
plugin entry point, called by pppd
----------------------------------------------------------------------------- */

int start(CFBundleRef ref)
{

    bundle = ref;

    CFRetain(bundle);

    // setup hooks
    acl_hook = dsaccess_authorize_user;
            
    info("Directory Services Authorization plugin initialized\n");
        
    return 0;
}


//----------------------------------------------------------------------
//	dsaccess_authorize_user
//----------------------------------------------------------------------
static int dsaccess_authorize_user(u_char* name, int len)
{

    tDirReference			dirRef;
    tDirStatus				dsResult = eDSNoErr;
    int						authorized = 0;
    tDirNodeReference 		searchNodeRef;
    tAttributeValueEntryPtr	gUID;
    UInt32					searchNodeCount;
    char*					user_name;
	uuid_t					userid;
	int						result;
	int						ismember;
    
    if (len < 1) {
        error("DSAccessControl plugin: invalid user name has zero length\n");
        return 0;
    }
    if ((user_name = (char*)malloc(len + 1)) == 0) {
        error("DSAccessControl plugin: unable to allocate memory for user name\n");
        return 0;
    }
    bcopy(name, user_name, len);
    *(user_name + len) = 0;

    if ((dsResult = dsOpenDirService(&dirRef)) == eDSNoErr) {  
        // get the search node ref
        if ((dsResult = dsauth_get_search_node_ref(dirRef, 1, &searchNodeRef, &searchNodeCount)) == eDSNoErr) {
            // get the user's generated user Id
            if ((dsResult = dsauth_get_user_attr(dirRef, searchNodeRef, user_name, kDS1AttrGeneratedUID, &gUID)) == eDSNoErr) {
                if (gUID != 0) {
					if (!mbr_string_to_uuid(gUID->fAttributeValueData.fBufferData, userid)) {
						// check if user is member authorized
						result = mbr_check_service_membership(userid, VPN_SERVICE_NAME, &ismember);
						if (result == ENOENT || (result == 0 && ismember != 0))
							authorized = 1;
					}
					dsDeallocAttributeValueEntry(dirRef, gUID);
                }
            }
            dsCloseDirNode(searchNodeRef);		// close the search node
        }
        dsCloseDirService(dirRef);
    }

    if (authorized)
        notice("DSAccessControl plugin: User '%s' authorized for access\n", user_name);
    else
        notice("DSAccessControl plugin: User '%s' not authorized for access\n", user_name);
    free(user_name);
    return authorized;
}

