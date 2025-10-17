/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 28, 2025.
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
#include "kuser_locl.h"
#include "kcc-commands.h"
#include "heimcred.h"
#include "heimbase.h"

/*
 *
 */

int
dump_credentials(struct dump_credentials_options *opt, int argc, char **argv)
{
    CFDictionaryRef query;
    CFArrayRef array;
    CFIndex n, count;
    
    const void *keys[] = { kHEIMAttrType };
    void *values[1] = { NULL };
    
    
    if (opt->type_string == NULL) {
	keys[0] = (void*)kHEIMObjectType;
	values[0] = (void *)kHEIMObjectAny;
    } else if (strcasecmp(opt->type_string, "Kerberos") == 0)
	values[0] = (void *)kHEIMTypeKerberos;
    else if (strcasecmp(opt->type_string, "NTLM") == 0)
	values[0] = (void *)kHEIMTypeNTLM;
    else if (strcasecmp(opt->type_string, "SCRAM") == 0)
	values[0] = (void *)kHEIMTypeSCRAM;
    else if (strcasecmp(opt->type_string, "Configuration") == 0)
	values[0] = (void *)kHEIMTypeConfiguration;
    else if (strcasecmp(opt->type_string, "Generic") == 0)
	values[0] = (void *)kHEIMTypeGeneric;
    else if (strcasecmp(opt->type_string, "Schema") == 0)
	values[0] = (void *)kHEIMTypeSchema;
    else {
	printf("unknown type; %s\n", opt->type_string);
	return 1;
    }
	
    query = CFDictionaryCreate(NULL, keys, (const void **)values, sizeof(keys)/sizeof(keys[0]), &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
    if (query == NULL)
	errx(1, "out of memory");

    array = HeimCredCopyQuery(query);
    CFRelease(query);
    if (array == NULL) {
	printf("no credentials\n");
	return 0;
    }
    
    count = CFArrayGetCount(array);
    for (n = 0; n < count; n++) {
	HeimCredRef cred = (HeimCredRef)CFArrayGetValueAtIndex(array, n);

	CFShow(cred);
	
	if (opt->verbose_flag) {
	    CFDictionaryRef attrs;

	    attrs = HeimCredCopyAttributes(cred, NULL, NULL);
	    if (attrs) {
		CFStringRef objType = CFDictionaryGetValue(attrs, kHEIMObjectType);
		if (!CFEqual(objType, kHEIMTypeKerberosAcquireCred)) {
		    CFShow(attrs);
		}
		CFRelease(attrs);
	    }
	}
	
    }

    return 0;
}
