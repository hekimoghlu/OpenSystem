/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 5, 2022.
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
#import <syslog.h>

#import <Heimdal/HeimdalSystemConfiguration.h>

#import <Foundation/Foundation.h>
#import <SystemConfiguration/SystemConfiguration.h>
#import <Heimdal/krb5.h>
#import <Heimdal/locate_plugin.h>

/**
 * Reachablity plugin reads System Configuration to pick up the realm
 * configuration from OpenDirectory plugins, both OD and AD.
 *
 * The keys published is:
 *
 * Kerberos:REALM = {
 *   kadmin = [ { host = "hostname", port = "port-number" } ]
 *   kdc = [ .. ]
 *   kpasswd = [ ]
 * }
 *
 * port is optional
 *
 * The following behaivor is expected:
 *
 * 1. Not joined to a domain
 *      no entry published
 * 2. Joined to a domain and replica AVAILABLE:
 *       entry pushlished with content
 * 3. Joined to a domain and replica UNAVAILABLE
 *       entry pushlished, but no content
 *
 */

static krb5_error_code
reachability_init(krb5_context context, void **ctx)
{
    *ctx = NULL;
    return 0;
}

static void
reachability_fini(void *ctx)
{
}

static krb5_error_code
reachability_lookup(void *ctx,
		    unsigned long flags,
		    enum locate_service_type service,
		    const char *realm,
		    int domain,
		    int type,
		    int (*addfunc)(void *,int,struct sockaddr *),
		    void *addctx)
{
    krb5_error_code ret;
    NSAutoreleasePool *pool;
    NSString *svc, *sckey, *host, *port;
    struct addrinfo hints, *ai0, *ai;
    SCDynamicStoreRef store = NULL;
    NSDictionary *top = NULL;
    NSArray *vals;
    NSString *defport = NULL;
    int found_entry = 0;
    id rp;
    
    @try {
	pool = [[NSAutoreleasePool alloc] init];

	switch(service) {
	case locate_service_kdc:
	case locate_service_master_kdc:
	case locate_service_krb524:
	    svc = (NSString *)HEIMDAL_SC_LOCATE_TYPE_KDC;
	    defport = @"88";
	    break;
	case locate_service_kpasswd:
	    svc = (NSString *)HEIMDAL_SC_LOCATE_TYPE_KPASSWD;
	    defport = @"464";
	    break;
	case locate_service_kadmin:
	    svc = (NSString *)HEIMDAL_SC_LOCATE_TYPE_ADMIN;
	    defport = @"749";
	    break;
	}
	if (defport == NULL) {
	    ret = KRB5_PLUGIN_NO_HANDLE;
	    goto out;
	}
	
	store = SCDynamicStoreCreate(kCFAllocatorDefault, CFSTR("Kerberos"), NULL, NULL);
	sckey = [NSString stringWithFormat:@"%@%s",
			  (NSString *)HEIMDAL_SC_LOCATE_REALM_PREFIX, realm];
	top = (NSDictionary *)SCDynamicStoreCopyValue(store, (CFStringRef)sckey);
	if (top == NULL) {
	    ret = KRB5_PLUGIN_NO_HANDLE;
	    goto out;
	}

	vals = [top valueForKey:svc];
	if (vals == NULL) {
	    ret = KRB5_PLUGIN_NO_HANDLE;
	    goto out;
	}

	if ([vals count] == 0)
	    syslog(LOG_WARNING,
		   "Kerberos-Reachability SystemConfiguration returned 0 entries for %s",
		   realm);
	
	for (NSDictionary *a in vals) {
	    host = [a valueForKey:(NSString *)HEIMDAL_SC_LOCATE_HOST];
	    
	    rp = [a valueForKey:(NSString *)HEIMDAL_SC_LOCATE_PORT];
	    if ([rp isKindOfClass:[NSString class]])
		port = rp;
	    else if ([rp respondsToSelector:@selector(stringValue)])
		port = [rp stringValue];
	    else
		port = defport;
	    if (port == nil)
		continue;
	    
	    memset(&hints, 0, sizeof(hints));
	    hints.ai_flags = 0;
	    hints.ai_family = type;
	    hints.ai_socktype = domain;
	    
	    if (getaddrinfo([host UTF8String], [port UTF8String], &hints, &ai0) != 0)
		continue;
	    
	    for (ai = ai0; ai != NULL; ai = ai->ai_next) {
		ret = addfunc(addctx, ai->ai_socktype, ai->ai_addr);
		if (ret == 0)
		    found_entry = 1;
	    }
	    freeaddrinfo(ai0);
	}
	
	if (!found_entry)
	    ret = KRB5_KDC_UNREACH;
	else
	    ret = 0;
     out:
	do {} while(0);
    }
    @catch (NSException * __unused exception) { }
    @finally {

	if (top)
	    CFRelease((CFTypeRef)top);
	if (store)
	    CFRelease(store);
	[pool drain];
    }

    return ret;
}

static krb5_error_code
reachability_lookup_old(void *ctx,
			enum locate_service_type service,
			const char *realm,
			int domain,
			int type,
			int (*addfunc)(void *,int,struct sockaddr *),
			void *addctx)
{
    return reachability_lookup(ctx, KRB5_PLF_ALLOW_HOMEDIR, service,
			       realm, domain, type, addfunc, addctx);
}

krb5plugin_service_locate_ftable service_locator = {
    KRB5_PLUGIN_LOCATE_VERSION_2,
    reachability_init,
    reachability_fini,
    reachability_lookup_old,
    reachability_lookup
};
