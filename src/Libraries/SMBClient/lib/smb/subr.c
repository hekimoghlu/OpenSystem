/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 7, 2024.
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
#include <mach/mach.h>
#include <mach/mach_error.h>
#include <servers/bootstrap.h>
#include <IOKit/kext/KextManager.h>

#include <netsmb/netbios.h>
#include <netsmb/smb_lib.h>
#include <netsmb/nb_lib.h>

#include <smbfs/smbfs.h>

#define CFENVFORMATSTRING "__CF_USER_TEXT_ENCODING=0x%X:0:0"

void smb_ctx_hexdump(const char *func, const char *s, unsigned char *buf, size_t inlen)
{
	char printstr[512];
	size_t maxlen;
	char *strPtr;
    int32_t addr, len = (int32_t)inlen;
    int32_t i;
	
	os_log_error(OS_LOG_DEFAULT, "%s: %s %p length %ld",  func, s, buf, inlen);
	if (buf == NULL) {
		return;
	}
    addr = 0;
    while( addr < len )
    {
		strPtr = printstr;
		maxlen = sizeof(printstr);
        strPtr += snprintf(strPtr, maxlen, "%6.6x - " , addr );
		maxlen -= (strPtr - printstr);
        for( i=0; i<16; i++ )
        {
            if( addr+i < len )
				strPtr += snprintf(strPtr, maxlen, "%2.2x ", buf[addr+i]);
            else
 				strPtr += snprintf(strPtr, maxlen, "   ");
			maxlen -= (strPtr - printstr);
       }
		strPtr += snprintf(strPtr, maxlen, " \"");
		maxlen -= (strPtr - printstr);
        for( i=0; i<16; i++ )
        {
            if( addr+i < len )
            {
                if(( buf[addr+i] > 0x19 ) && ( buf[addr+i] < 0x7e ) )
					strPtr += snprintf(strPtr, maxlen, "%c", buf[addr+i] );
                else
					strPtr += snprintf(strPtr, maxlen, ".");
				maxlen -= (strPtr - printstr);
            }
        }
		os_log_error(OS_LOG_DEFAULT, "%s", printstr);
        addr += 16;
    }
	os_log_error(OS_LOG_DEFAULT, " ");
}

/*
 * Load our kext
 */
int smb_load_library(void)
{
	struct vfsconf vfc;
	kern_return_t status;
	
	setlocale(LC_CTYPE, "");
	if (getvfsbyname(SMBFS_VFSNAME, &vfc) != 0) {
		/* Need to load the kext */
		status = KextManagerLoadKextWithIdentifier(CFSTR("com.apple.filesystems.smbfs") ,NULL);
        if (status != KERN_SUCCESS) {
			os_log_error(OS_LOG_DEFAULT, "Loading com.apple.filesystems.smbfs status = 0x%x", 
						 status);
			return EIO;
        }
	}
	return 0;
}
