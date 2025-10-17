/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 16, 2024.
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
#import <err.h>
#import <stdio.h>
#import <arpa/inet.h>
#import <netdb.h>
#import <sys/param.h>
#import <sys/socket.h>

#import <Foundation/Foundation.h>
#import <SystemConfiguration/SystemConfiguration.h>
#import <Heimdal/HeimdalSystemConfiguration.h>
#import <Heimdal/krb5.h>
#import <Heimdal/config_plugin.h>

/**
 * Configuration plugin uses configuration in SC for Kerberos
 */

struct config_ctx {
    SCDynamicStoreRef store;
};


static krb5_error_code
get_default_realm(krb5_context context, void *ptr, void *userctx,
		  void (*add_realms)(krb5_context, void *, krb5_const_realm))
{
    krb5_error_code ret = KRB5_PLUGIN_NO_HANDLE;
    struct config_ctx *ctx = ptr;
    NSAutoreleasePool *pool;
    NSArray *vals = NULL;
    
    @try {
	pool = [[NSAutoreleasePool alloc] init];

	vals = (NSArray *)SCDynamicStoreCopyValue(ctx->store, HEIMDAL_SC_DEFAULT_REALM);
	if (vals == NULL)
	    goto out;
	
	if ([vals count] == 0)
	    goto out;
	
	for (NSString *a in vals)
	    add_realms(context, userctx, [a UTF8String]);

	ret = 0;
    out:
	do { } while(0);
    }
    @catch (NSException * __unused exception) { }
    @finally {

	if (vals)
	    CFRelease((CFTypeRef)vals);
	[pool drain];
    }

    return ret;
}

static krb5_error_code
get_host_domain(krb5_context context, const char *hostname, void *ptr, void *userptr,
		void (*add_realms)(krb5_context, void *, krb5_const_realm))
{
    krb5_error_code ret = KRB5_PLUGIN_NO_HANDLE;
    struct config_ctx *ctx = ptr;
    NSAutoreleasePool *pool;
    NSArray *vals = NULL;
    
    @try {
	pool = [[NSAutoreleasePool alloc] init];
	NSMutableArray *res = [NSMutableArray arrayWithCapacity:0];
	NSString *host = [NSString stringWithUTF8String:hostname];
	
	vals = (NSArray *)SCDynamicStoreCopyValue(ctx->store, HEIMDAL_SC_DOMAIN_REALM_MAPPING);
	if (vals == NULL)
	    goto out;
	
	/* search dict for matches, all matches from first domain that matches */
	for (NSDictionary *a in vals) {
	    for (NSString *domain in a)
		if ([host hasSuffix:domain])
		    [res addObject:[a valueForKey:domain]];
	    
	    if ([res count])
		break;
	}
	if ([res count] == 0)
	    goto out;

	for (NSString *realm in res)
	    add_realms(context, userptr, [realm UTF8String]);
	
	ret = 0;
	out:
	do { } while(0);
    }
    @catch (NSException * __unused exception) { }
    @finally {
	
	if (vals)
	    CFRelease((CFTypeRef)vals);
	[pool drain];
    }
    
    return ret;
}

static krb5_error_code
config_init(krb5_context context, void **ptr)
{
    struct config_ctx *ctx = calloc(1, sizeof(*ctx));

    if (ctx == NULL)
	return ENOMEM;

    ctx->store = SCDynamicStoreCreate(kCFAllocatorDefault, CFSTR("SCKerberosConfig"), NULL, NULL);
    if (ctx->store == NULL) {
	free(ctx);
	return ENOMEM;
    }
	
    *ptr = ctx;
    return 0;
}

static void
config_fini(void *ptr)
{
    struct config_ctx *ctx = ptr;

    CFRelease(ctx->store);
    free(ctx);
}


krb5plugin_config_ftable krb5_configuration = {
    KRB5_PLUGIN_CONFIGURATION_VERSION_1,
    config_init,
    config_fini,
    get_default_realm,
    get_host_domain
};
