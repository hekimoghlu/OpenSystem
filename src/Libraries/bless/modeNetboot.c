/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 18, 2024.
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
 *  modeNetboot.c
 *  bless
 *
 *  Created by Shantonu Sen on 10/10/05.
 *  Copyright 2005-2007 Apple Inc. All Rights Reserved.
 *
 */

#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/socket.h>
#include <net/if.h>
#include <arpa/nameser.h>
#include <netinet/in.h>
#include <arpa/inet.h>


#include "enums.h"
#include "structs.h"

#include "bless.h"
#include "bless_private.h"
#include "protos.h"

static bool validateAddress(const char *host);

static int parseURL(BLContextPtr context,
                    const char *url,
                    char * scheme,
                    char * interface,
                    char * host,
                    char * path,
                    bool   requiresPath,
                    const char *requiresScheme,
                    bool	useBackslash);


int modeNetboot(BLContextPtr context, struct clarg actargs[klast])
{
    int ret;
	BLPreBootEnvType	preboot = getPrebootType();
    char                interface[IF_NAMESIZE];
    char                host[NS_MAXDNAME];
    char                path[MAXPATHLEN];
	char				scheme[128];
	bool				useBackslash = false;
    
    if(!( actargs[kserver].present || actargs[kbooter].present)) {
        blesscontextprintf(context, kBLLogLevelError,
                           "No NetBoot server specification provided\n");
        return 1;
    }
    
	if(preboot == kBLPreBootEnvType_OpenFirmware) {
		useBackslash = true;
	}
	
	
    if (actargs[kserver].present) {
        blesscontextprintf(context, kBLLogLevelVerbose,
                           "NetBoot server specification given as: %s\n",
                           actargs[kserver].argument);
        
        ret = parseURL(context,
                       actargs[kserver].argument,
					   scheme,
                       interface,
                       host,
                       path,
                       false,
                       NULL,
					   useBackslash);
        
    } else {
        blesscontextprintf(context, kBLLogLevelVerbose,
                           "NetBoot booter given as: %s\n",
                           actargs[kbooter].argument);  
        
        ret = parseURL(context,
                       actargs[kbooter].argument,
					   scheme,
                       interface,
                       host,
                       path,
                       true,
                       "tftp",
					   useBackslash);
        
    }
    
    if(ret)
        return 1;
    
    if(preboot == kBLPreBootEnvType_OpenFirmware) {
        // XXX temporary stub
#define NVRAM "/usr/sbin/nvram"
        char * OFSettings[6];
        
        char bootdevice[1024];
        char bootfile[1024];
        char bootcommand[1024];
        char bootargs[1024];
		char ofstring[1024];
        
        pid_t p;
        int status;
        
		if (0 != strcmp(scheme, "bsdp") && 0 != strcmp(scheme, "tftp")) {
			blesscontextprintf(context, kBLLogLevelError,
                               "Netboot scheme %s not supported on Open Firmware systems\n",
							   scheme);  
		}
		
		ret = BLGetOpenFirmwareBootDeviceForNetworkPath(context,
                                                        interface,
                                                        strcmp(host, "255.255.255.255") == 0 ? NULL : host,
                                                        path,
                                                        ofstring,
                                                        sizeof(ofstring));
		if(ret)
			return 3;
        
		snprintf(bootdevice, sizeof(bootdevice), "boot-device=%s", ofstring);
        
        if(actargs[kkernel].present) {
            ret = parseURL(context,
                           actargs[kkernel].argument,
						   scheme,
                           interface,
                           host,
                           path,
                           true,
                           "tftp",
						   useBackslash);            
            if(ret)
                return 1;
            
			ret = BLGetOpenFirmwareBootDeviceForNetworkPath(context,
                                                            interface,
                                                            strcmp(host, "255.255.255.255") == 0 ? NULL : host,
                                                            path,
                                                            ofstring,
                                                            sizeof(ofstring));
			if(ret)
				return 4;
            
            snprintf(bootfile, sizeof(bootfile), "boot-file=%s", ofstring);
        } else {
            snprintf(bootfile, sizeof(bootfile), "boot-file=");
        }
        
        if(actargs[kmkext].present) {
            blesscontextprintf(context, kBLLogLevelError,
                               "mkext option not supported on Open Firmware systems\n");  
            return 5;
        }
        
        strlcpy(bootcommand, "boot-command=mac-boot", sizeof(bootcommand));
        strlcpy(bootargs, "boot-args=", sizeof(bootargs));
        
        OFSettings[0] = NVRAM;
        OFSettings[1] = bootdevice;
        OFSettings[2] = bootfile;
        OFSettings[3] = bootcommand;
        OFSettings[4] = bootargs;
        OFSettings[5] = NULL;
        
        blesscontextprintf(context, kBLLogLevelVerbose,  "OF Setings:\n" );    
        blesscontextprintf(context, kBLLogLevelVerbose,  "\t\tprogram: %s\n", OFSettings[0] );
        blesscontextprintf(context, kBLLogLevelVerbose,  "\t\t%s\n", OFSettings[1] );
        blesscontextprintf(context, kBLLogLevelVerbose,  "\t\t%s\n", OFSettings[2] );
        blesscontextprintf(context, kBLLogLevelVerbose,  "\t\t%s\n", OFSettings[3] );
        blesscontextprintf(context, kBLLogLevelVerbose,  "\t\t%s\n", OFSettings[4] );
        
        p = fork();
        if (p == 0) {
            int ret = execv(NVRAM, OFSettings);
            if(ret == -1) {
                blesscontextprintf(context, kBLLogLevelError,  "Could not exec %s\n", NVRAM );
            }
            _exit(1);
        }
        
        do {
            p = wait(&status);
        } while (p == -1 && errno == EINTR);
        
        if(p == -1 || status) {
            blesscontextprintf(context, kBLLogLevelError,  "%s returned non-0 exit status\n", NVRAM );
            return 3;
        }
        
    } else if(preboot == kBLPreBootEnvType_EFI) {
        CFStringRef booterXML = NULL, kernelXML = NULL, mkextXML = NULL, kernelcacheXML = NULL;
		BLNetBootProtocolType protocol = kBLNetBootProtocol_Unknown;
        
		if (0 == strcmp(scheme, "bsdp") || 0 == strcmp(scheme, "tftp")) {
			protocol = kBLNetBootProtocol_BSDP;
		} else if (0 == strcmp(scheme, "pxe")) {
			protocol = kBLNetBootProtocol_PXE;
            
			if (0 != strcmp(host, "255.255.255.255")) {
				blesscontextprintf(context, kBLLogLevelError,
								   "PXE requires broadcast address 255.255.255.255\n");  
				return 3;				
			}
			
		} else {			
			blesscontextprintf(context, kBLLogLevelError,
                               "Netboot scheme %s not supported on EFI systems\n",
							   scheme);  
			return 3;
		}
		
        
        ret = BLCreateEFIXMLRepresentationForNetworkPath(context,
														 protocol,
                                                         interface,
                                                         strcmp(host, "255.255.255.255") == 0 ? NULL : host,
                                                         strlen(path) > 0 ? path : NULL,
                                                         actargs[koptions].present ? actargs[koptions].argument : NULL,
                                                         &booterXML);
        if(ret)
            return 3;
        
        
        if(actargs[kkernel].present) {
            ret = parseURL(context,
                           actargs[kkernel].argument,
						   scheme,
                           interface,
                           host,
                           path,
                           true,
                           "tftp",
						   false);            
            if(ret)
                return 1;
            
            ret = BLCreateEFIXMLRepresentationForNetworkPath(context,
															 protocol,
                                                             interface,
                                                             host,
                                                             path,
                                                             NULL,
                                                             &kernelXML);
            if(ret)
                return 3;
        }
        
        
        if(actargs[kmkext].present) {
            ret = parseURL(context,
                           actargs[kmkext].argument,
						   scheme,
                           interface,
                           host,
                           path,
                           true,
                           "tftp",
						   false);            
            if(ret)
                return 1;
            
            ret = BLCreateEFIXMLRepresentationForNetworkPath(context,
															 protocol,
                                                             interface,
                                                             host,
                                                             path,
                                                             NULL,
                                                             &mkextXML);
            if(ret)
                return 3;
        }

        if(actargs[kkernelcache].present) {
            ret = parseURL(context,
                           actargs[kkernelcache].argument,
						   scheme,
                           interface,
                           host,
                           path,
                           true,
                           "tftp",
						   false);            
            if(ret)
                return 1;
            
            ret = BLCreateEFIXMLRepresentationForNetworkPath(context,
															 protocol,
                                                             interface,
                                                             host,
                                                             path,
                                                             NULL,
                                                             &kernelcacheXML);
            if(ret)
                return 3;
        }
        
        
        ret = setefinetworkpath(context, booterXML, kernelXML, mkextXML, kernelcacheXML,
                                actargs[knextonly].present);
        
		if(ret) {
			blesscontextprintf(context, kBLLogLevelError,  "Can't set EFI\n" );
			return 1;
		} else {
			blesscontextprintf(context, kBLLogLevelVerbose,  "EFI set successfully\n" );
		}
    } else {
        blesscontextprintf(context, kBLLogLevelError,
                           "NetBoot not supported on this system\n");
        return 3;        
    }
    
    return 0;
}

static bool validateAddress(const char *host)
{
    in_addr_t addr;
    int ret;
    
    ret = inet_pton(PF_INET, host, &addr);
    if(ret == 1)
        return true;
    else
        return false;
}

static int parseURL(BLContextPtr context,
                    const char *url,
					char * scheme,
                    char * interface,
                    char * host,
                    char * path,
                    bool   requiresPath,
                    const char *requiresScheme,
					bool	useBackslash)
{
    int                 ret;
    
    CFURLRef            serverURL;
    CFStringRef         schemeString, interfaceString, pathString, hostString;
    CFStringRef         requiredScheme = NULL;
    
    serverURL = CFURLCreateAbsoluteURLWithBytes(kCFAllocatorDefault,
                                                (const UInt8 *)url,
                                                strlen(url),
                                                kCFStringEncodingUTF8,
                                                NULL, false);
    
    if(!serverURL || !CFURLCanBeDecomposed(serverURL)) {
        if(serverURL) CFRelease(serverURL);
        
        blesscontextprintf(context, kBLLogLevelError,
                           "Could not interpret %s as a URL\n", url);
        return 2;        
    }
    
    schemeString = CFURLCopyScheme(serverURL);
    if (requiresScheme) {
        requiredScheme = CFStringCreateWithCString(kCFAllocatorDefault,requiresScheme,kCFStringEncodingUTF8);        
    }
    if(!schemeString || (requiredScheme && !CFEqual(schemeString, requiredScheme))) {
        
        blesscontextprintf(context, kBLLogLevelError,
                           "Unrecognized scheme %s\n", BLGetCStringDescription(schemeString));

        if(schemeString) CFRelease(schemeString);
        if(requiredScheme) CFRelease(requiredScheme);

        return 2;               
    }
    
	if(!CFStringGetCString(schemeString,scheme,128,kCFStringEncodingUTF8)) {
		blesscontextprintf(context, kBLLogLevelError,
						   "Can't interpret scheme as string\n");
		return 3;            
	}		
	
    if(requiredScheme) CFRelease(requiredScheme);
    CFRelease(schemeString);
    
    interfaceString = CFURLCopyUserName(serverURL);
    
    if(interfaceString == NULL) {
        
        ret = BLGetPreferredNetworkInterface(context, interface);
        if(ret) {
            blesscontextprintf(context, kBLLogLevelError,
                               "Failed to determine preferred network interface\n");
            return 1;            
        } else {
            blesscontextprintf(context, kBLLogLevelVerbose,
                               "Preferred network interface is %s\n", interface);            
        }
    } else {
        if(!CFStringGetCString(interfaceString,interface,IF_NAMESIZE,kCFStringEncodingUTF8)) {
            CFRelease(interfaceString);
            blesscontextprintf(context, kBLLogLevelError,
                               "Can't interpret interface as string\n");
            return 3;            
        }
        
        if(!BLIsValidNetworkInterface(context, interface)) {
            blesscontextprintf(context, kBLLogLevelError,
                               "%s is not a valid interface\n", interface);            
            return 4;
        }
        
        CFRelease(interfaceString);
    }
    
    blesscontextprintf(context, kBLLogLevelVerbose,
                       "Using interface %s\n", interface);            
    
    pathString = CFURLCopyStrictPath(serverURL, NULL);
    
    // path component must be NULL or empty
    if(requiresPath) {
        if(pathString == NULL || CFEqual(pathString, CFSTR(""))) {
            if(pathString) CFRelease(pathString);
            blesscontextprintf(context, kBLLogLevelError,
                               "Specification must contain a path\n");
            return 5;        
        }
        if(!CFStringGetCString(pathString,path,MAXPATHLEN,kCFStringEncodingUTF8)) {
            CFRelease(pathString);
            blesscontextprintf(context, kBLLogLevelError,
                               "Can't interpret path as string\n");
            return 5;            
        }        
    } else {
        if(!(pathString == NULL || CFEqual(pathString, CFSTR("")))) {
            CFRelease(pathString);
            blesscontextprintf(context, kBLLogLevelError,
                               "Specification can't contain a path\n");
            return 5;        
        }
        path[0] = '\0';
    }
	
    if(pathString) CFRelease(pathString);
    
	if(useBackslash) {
		size_t i, len;
		for(i=0, len=strlen(path); i < len; i++) {
			if(path[i] == '/') {
				path[i] = '\\';
			}
		}
	}
	
	
    hostString = CFURLCopyHostName(serverURL);
    
    // host must be present
    if(hostString == NULL || CFEqual(hostString, CFSTR(""))) {
        if(hostString) CFRelease(hostString);
        blesscontextprintf(context, kBLLogLevelError,
                           "Specification doesn't contain host\n");
        return 5;        
    }
    
    if(!CFStringGetCString(hostString,host,NS_MAXDNAME,kCFStringEncodingUTF8)) {
        CFRelease(hostString);
        blesscontextprintf(context, kBLLogLevelError,
                           "Can't interpret host as string\n");
        return 5;            
    }
    
    CFRelease(hostString);
    
    if(!validateAddress(host)) {
        blesscontextprintf(context, kBLLogLevelError,
                           "Can't interpret host %s as an IPv4 address\n",
                           host);
        return 6;
    }
    
    return 0;
}
