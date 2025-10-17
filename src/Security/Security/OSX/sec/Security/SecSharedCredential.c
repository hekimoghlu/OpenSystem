/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 24, 2024.
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
#include <Security/SecSharedCredential.h>
#include <Security/SecBasePriv.h>
#include <Security/SecEntitlements.h>
#include <utilities/SecCFError.h>
#include <utilities/SecCFWrappers.h>
#include "SecItemInternal.h"
#include <ipc/securityd_client.h>
#include "SecPasswordGenerate.h"

/* forward declarations */
OSStatus SecAddSharedWebCredentialSync(CFStringRef fqdn, CFStringRef account, CFStringRef password, CFErrorRef *error);
OSStatus SecCopySharedWebCredentialSync(CFStringRef fqdn, CFStringRef account, CFArrayRef *credentials, CFErrorRef *error);
CFStringRef SecCopyFQDNFromEntitlementString(CFStringRef entitlement);

#if SHAREDWEBCREDENTIALS

// OSX now has SWC enabled, but cannot link SharedWebCredentials framework: rdar://59958701
#if TARGET_OS_OSX || TARGET_OS_MACCATALYST

OSStatus SecAddSharedWebCredentialSync(CFStringRef fqdn,
    CFStringRef account,
    CFStringRef password,
    CFErrorRef *error)
{
    OSStatus status = errSecUnimplemented;
    if (error) {
        SecError(status, error, CFSTR("SecAddSharedWebCredentialSync not supported on this platform"));
    }
    return status;
}

#else

OSStatus SecAddSharedWebCredentialSync(CFStringRef fqdn,
    CFStringRef account,
    CFStringRef password,
    CFErrorRef *error)
{
    OSStatus status;
    __block CFErrorRef* outError = error;
    __block CFMutableDictionaryRef args = CFDictionaryCreateMutable(kCFAllocatorDefault,
        0, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
    if (fqdn) {
        CFDictionaryAddValue(args, kSecAttrServer, fqdn);
    }
    if (account) {
        CFDictionaryAddValue(args, kSecAttrAccount, account);
    }
    if (password) {
        CFDictionaryAddValue(args, kSecSharedPassword, password);
    }
    status = SecOSStatusWith(^bool (CFErrorRef *error) {
        CFTypeRef raw_result = NULL;
        bool xpc_result = false;
        bool internal_spi = false; // TODO: support this for SecurityDevTests
        if(internal_spi && gSecurityd && gSecurityd->sec_add_shared_web_credential) {
            xpc_result = gSecurityd->sec_add_shared_web_credential(args, NULL, NULL, NULL, SecAccessGroupsGetCurrent(), &raw_result, error);
        } else {
            xpc_result = cftype_client_to_bool_cftype_error_request(sec_add_shared_web_credential_id, args, SecSecurityClientGet(), &raw_result, error);
        }
        CFReleaseSafe(args);
        if (!xpc_result) {
            if (NULL == *error) {
                SecError(errSecInternal, error, CFSTR("Internal error (XPC failure)"));
            }
        }
        if (outError) {
            *outError = (error) ? *error : NULL;
            CFRetainSafe(*outError);
        } else {
            CFReleaseNull(*error);
        }
        CFReleaseNull(raw_result);
        return xpc_result;
    });

    return status;
}
#endif /* !TARGET_OS_OSX || !TARGET_OS_MACCATALYST */
#endif /* SHAREDWEBCREDENTIALS */

void SecAddSharedWebCredential(CFStringRef fqdn,
    CFStringRef account,
    CFStringRef password,
    void (^completionHandler)(CFErrorRef error))
{
	__block CFErrorRef error = NULL;
	__block dispatch_queue_t dst_queue = dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT,0);
#if SHAREDWEBCREDENTIALS

    /* type check input arguments */
	CFStringRef errStr = NULL;
	if (!fqdn || CFGetTypeID(fqdn) != CFStringGetTypeID() || !CFStringGetLength(fqdn) ||
		!account || CFGetTypeID(account) != CFStringGetTypeID() || !CFStringGetLength(account) ) {
		errStr = CFSTR("fqdn or account was not of type CFString, or not provided");
	}
	else if (password && CFGetTypeID(password) != CFStringGetTypeID()) {
		errStr = CFSTR("non-nil password was not of type CFString");
	}
	if (errStr) {
		SecError(errSecParam, &error, CFSTR("%@"), errStr);
		dispatch_async(dst_queue, ^{
			if (completionHandler) {
				completionHandler(error);
			}
			CFReleaseSafe(error);
		});
		return;
	}

	__block CFStringRef serverStr = CFRetainSafe(fqdn);
	__block CFStringRef accountStr = CFRetainSafe(account);
	__block CFStringRef passwordStr = CFRetainSafe(password);

	dispatch_async(dst_queue, ^{
		OSStatus status = SecAddSharedWebCredentialSync(serverStr, accountStr, passwordStr, &error);
		CFReleaseSafe(serverStr);
		CFReleaseSafe(accountStr);
		CFReleaseSafe(passwordStr);

		if (status && !error) {
			SecError(status, &error, CFSTR("Error adding shared password"));
		}
		dispatch_async(dst_queue, ^{
			if (completionHandler) {
				completionHandler(error);
			}
			CFReleaseSafe(error);
		});
	});
#else
    SecError(errSecParam, &error, CFSTR("SharedWebCredentials not supported on this platform"));
    dispatch_async(dst_queue, ^{
        if (completionHandler) {
            completionHandler(error);
        }
        CFReleaseSafe(error);
    });
#endif
}

void SecRequestSharedWebCredential(CFStringRef fqdn,
    CFStringRef account,
    void (^completionHandler)(CFArrayRef credentials, CFErrorRef error))
{
    __block CFErrorRef error = NULL;
    __block dispatch_queue_t dst_queue = dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT,0);
#if SHAREDWEBCREDENTIALS
    __block CFArrayRef result = NULL;
    __block CFStringRef serverStr = CFRetainSafe(fqdn);
    __block CFStringRef accountStr = CFRetainSafe(account);

    /* type check input arguments */
    CFStringRef errStr = NULL;
    if (fqdn && (CFGetTypeID(fqdn) != CFStringGetTypeID() || !CFStringGetLength(fqdn))) {
        errStr = CFSTR("fqdn was empty or not a CFString");
    }
    /* a NULL 'fqdn' is documented to implicitly specify the domain(s) in
       the 'com.apple.developer.associated-domains' entitlement. Authentication
       Services doesn't pass the associated domain back to us, so we need to
       extract it ourselves, and then include it in the credentials dictionary
       that we pass to the completion handler. */
    if (!fqdn) {
        CFArrayRef domains = NULL;
        SecTaskRef task = SecTaskCreateFromSelf(NULL);
        if (task) {
            domains = (CFArrayRef)SecTaskCopyValueForEntitlement(task, kSecEntitlementAssociatedDomains, NULL);
        }
        CFIndex idx, count = (domains) ? CFArrayGetCount(domains) : 0;
        for (idx=0; idx < count; idx++) {
            CFStringRef str = (CFStringRef) CFArrayGetValueAtIndex(domains, idx);
            if ((serverStr = SecCopyFQDNFromEntitlementString(str)) != NULL) {
                break;
            }
        }
        CFReleaseSafe(domains);
        CFReleaseSafe(task);
    }
    if (!errStr && !serverStr) {
        errStr = CFSTR("fqdn was NULL, and no associated domains found");
    }
    if (!errStr && serverStr && (CFGetTypeID(serverStr) != CFStringGetTypeID() || !CFStringGetLength(serverStr))) {
        errStr = CFSTR("fqdn was empty or not a CFString");
    }
    if (!errStr && accountStr && (CFGetTypeID(accountStr) != CFStringGetTypeID() || !CFStringGetLength(accountStr))) {
        errStr = CFSTR("account was empty or not a CFString");
    }
    if (errStr) {
        CFReleaseSafe(serverStr);
        CFReleaseSafe(accountStr);
        SecError(errSecParam, &error, CFSTR("%@"), errStr);
        dispatch_async(dst_queue, ^{
            if (completionHandler) {
                completionHandler(result, error);
            }
            CFReleaseSafe(error);
            CFReleaseSafe(result);
        });
        return;
    }

    dispatch_async(dst_queue, ^{
		OSStatus status = SecCopySharedWebCredentialSync(serverStr, accountStr, &result, &error);
		CFReleaseSafe(serverStr);
		CFReleaseSafe(accountStr);

		if (status && !error) {
			SecError(status, &error, CFSTR("Error copying shared password"));
		}
		dispatch_async(dst_queue, ^{
			if (completionHandler) {
				completionHandler(result, error);
			}
			CFReleaseSafe(error);
			CFReleaseSafe(result);
		});
	});
#else
    SecError(errSecParam, &error, CFSTR("SharedWebCredentials not supported on this platform"));
    dispatch_async(dst_queue, ^{
        if (completionHandler) {
            completionHandler(NULL, error);
        }
        CFReleaseSafe(error);
    });
#endif /* SHAREDWEBCREDENTIALS */

}

CFStringRef SecCreateSharedWebCredentialPassword(void)
{

    CFStringRef password = NULL;
    CFErrorRef error = NULL;
    CFMutableDictionaryRef passwordRequirements = NULL;

    CFStringRef allowedCharacters = CFSTR("abcdefghkmnopqrstuvwxyzABCDEFGHJKLMNPQRSTUVWXYZ23456789");
    CFCharacterSetRef requiredCharactersLower = CFCharacterSetCreateWithCharactersInString(NULL, CFSTR("abcdefghkmnopqrstuvwxyz"));
    CFCharacterSetRef requiredCharactersUppder = CFCharacterSetCreateWithCharactersInString(NULL, CFSTR("ABCDEFGHJKLMNPQRSTUVWXYZ"));
    CFCharacterSetRef requiredCharactersNumbers = CFCharacterSetCreateWithCharactersInString(NULL, CFSTR("3456789"));

    int groupSize = 3;
    int groupCount = 4;
    int totalLength = (groupSize * groupCount);
    CFNumberRef groupSizeRef = CFNumberCreate(NULL, kCFNumberIntType, &groupSize);
    CFNumberRef groupCountRef = CFNumberCreate(NULL, kCFNumberIntType, &groupCount);
    CFNumberRef totalLengthRef = CFNumberCreate(NULL, kCFNumberIntType, &totalLength);
    CFStringRef separator = CFSTR("-");

    CFMutableArrayRef requiredCharacterSets = CFArrayCreateMutable(NULL, 0, &kCFTypeArrayCallBacks);
    CFArrayAppendValue(requiredCharacterSets, requiredCharactersLower);
    CFArrayAppendValue(requiredCharacterSets, requiredCharactersUppder);
    CFArrayAppendValue(requiredCharacterSets, requiredCharactersNumbers);

    passwordRequirements = CFDictionaryCreateMutable(NULL, 0, NULL, NULL);
    CFDictionaryAddValue(passwordRequirements, kSecPasswordAllowedCharactersKey, allowedCharacters);
    CFDictionaryAddValue(passwordRequirements, kSecPasswordRequiredCharactersKey, requiredCharacterSets);
    CFDictionaryAddValue(passwordRequirements, kSecPasswordGroupSize, groupSizeRef );
    CFDictionaryAddValue(passwordRequirements, kSecPasswordNumberOfGroups, groupCountRef);
    CFDictionaryAddValue(passwordRequirements, kSecPasswordSeparator, separator);
    CFDictionaryAddValue(passwordRequirements, kSecPasswordMaxLengthKey, totalLengthRef);
    CFDictionaryAddValue(passwordRequirements, kSecPasswordMinLengthKey, totalLengthRef);
    CFDictionaryAddValue(passwordRequirements, kSecPasswordDefaultForType, CFSTR("false"));
    CFRelease(requiredCharactersLower);
    CFRelease(requiredCharactersUppder);
    CFRelease(requiredCharactersNumbers);
    CFRelease(groupSizeRef);
    CFRelease(groupCountRef);
    CFRelease(totalLengthRef);

    password = SecPasswordGenerate(kSecPasswordTypeSafari, &error, passwordRequirements);

    CFRelease(requiredCharacterSets);
    CFRelease(passwordRequirements);
    if ((error && error != errSecSuccess) || !password)
    {
        if (password) CFRelease(password);
        secwarning("SecPasswordGenerate failed to generate a password for SecCreateSharedWebCredentialPassword.");
        return NULL;
    } else {
        return password;
    }

}
