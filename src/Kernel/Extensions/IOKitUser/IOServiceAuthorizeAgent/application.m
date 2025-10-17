/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 3, 2021.
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
#include "application.h"

#include <libproc.h>
#include <CoreFoundation/CFBundlePriv.h>
#include <Security/Security.h>

dispatch_queue_t __ApplicationGetQueue( void )
{
    static dispatch_once_t once;
    static dispatch_queue_t queue;

    dispatch_once( &once, ^
    {
        queue = dispatch_queue_create( 0, 0 );
    } );

    return queue;
}

CFBundleRef _ApplicationCopyBundle( pid_t processID )
{
    __block CFBundleRef bundle = 0;

    dispatch_sync( __ApplicationGetQueue( ), ^
    {
        int length;
        uint8_t path[ PROC_PIDPATHINFO_MAXSIZE ];

        length = proc_pidpath( processID, path, PROC_PIDPATHINFO_MAXSIZE );

        if ( length )
        {
            CFURLRef url;

            url = CFURLCreateFromFileSystemRepresentation( kCFAllocatorDefault, path, length, TRUE );

            if ( url )
            {
                bundle = _CFBundleCreateWithExecutableURLIfMightBeBundle( kCFAllocatorDefault, url );

                CFRelease( url );
            }
        }
    } );

    return bundle;
}

CFStringRef _ApplicationCopyIdentifier( pid_t processID, const audit_token_t *auditToken )
{
    __block CFStringRef identifier = 0;

    dispatch_sync( __ApplicationGetQueue( ), ^
    {
        CFMutableDictionaryRef attributes;

        attributes = CFDictionaryCreateMutable( kCFAllocatorDefault, 0, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks );

        if ( attributes )
        {
            CFNumberRef number;

            number = CFNumberCreate( kCFAllocatorDefault, kCFNumberIntType, &processID );

            if ( number )
            {
                CFDataRef auditTokenData;

                auditTokenData = CFDataCreate(NULL, (const UInt8 *) auditToken, sizeof(*auditToken));

                if ( auditTokenData )
                {
                    SecCodeRef code = 0;

                    CFDictionarySetValue( attributes, kSecGuestAttributePid, number );
                    CFDictionarySetValue( attributes, kSecGuestAttributeAudit, auditTokenData );

                    OSStatus status = SecCodeCopyGuestWithAttributes( 0, attributes, kSecCSDefaultFlags, &code );
                    if ( status == noErr )
                    {
						status = SecCodeCheckValidity(code, kSecCSDefaultFlags, NULL);
					}
                    if ( status == noErr )
                    {
                        CFDictionaryRef information = 0;

                        SecCodeCopySigningInformation( code, kSecCSDefaultFlags, &information );

                        if ( information )
                        {
                            identifier = CFDictionaryGetValue( information, kSecCodeInfoIdentifier );

                            if ( identifier )
                            {
                                CFRetain( identifier );
                            }

                            CFRelease( information );
                        }

                        CFRelease( code );
                    }

                    CFRelease(auditTokenData);
                }

                CFRelease( number );
            }

            CFRelease( attributes );
        }
    } );

    return identifier;
}
