/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 13, 2024.
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
#if OCTAGON

#import <AuthKit/AuthKit.h>

#import "keychain/ot/OTDefines.h"
#import "keychain/ot/ErrorUtils.h"
#import "keychain/ckks/CKKS.h"

/*
 * These are stolen from cloudkit to avoid having to link it (which we cannot do on darinwos).
 */

static NSString * const _CKErrorDomain = @"CKErrorDomain";
static NSString * const _CKUnderlyingErrorDomain = @"CKInternalErrorDomain";
static NSString * const _CKErrorRetryAfterKey = @"CKRetryAfter";
static NSString * const _CKPartialErrorsByItemIDKey   = @"CKPartialErrors";

static NSString * const _AKAppleIDAuthenticationErrorDomain = @"AKAuthenticationError";

enum {
    /*! Some items failed, but the operation succeeded overall. Check CKPartialErrorsByItemIDKey in the userInfo dictionary for more details.
     *  This error is only returned from CKOperation completion blocks, which are deprecated in swift.
     *  It will not be returned from (swift-only) CKOperation result blocks, which are their replacements
     */
    _CKErrorPartialFailure                 = 2,

    /*! Network not available */
    _CKErrorNetworkUnavailable             = 3,

    /*! Network error (available but CFNetwork gave us an error) */
    _CKErrorNetworkFailure                 = 4,

    /*! Client is being rate limited */
    _CKErrorRequestRateLimited             = 7,

    /*! The server rejected this request. This is a non-recoverable error */
    _CKErrorServerRejectedRequest          = 15,

    /* Request server errors */
    _CKErrorInternalServerInternalError = 2000,

    /* More Other */
    _CKErrorInternalPluginError = 6000,
};

@implementation NSError (Octagon)

- (BOOL)_isCKServerInternalError {
    NSError* underlyingError = self.userInfo[NSUnderlyingErrorKey];

    return [self.domain isEqualToString:_CKErrorDomain] &&
        self.code == _CKErrorServerRejectedRequest &&
        underlyingError &&
        [underlyingError.domain isEqualToString:_CKUnderlyingErrorDomain] &&
        underlyingError.code == _CKErrorInternalServerInternalError;
}

- (BOOL)isCuttlefishError:(CuttlefishErrorCode)cuttlefishErrorCode
{
    NSError *error = self;

    if ([error.domain isEqualToString:_CKErrorDomain] && error.code == _CKErrorServerRejectedRequest) {
        NSError* underlyingError = error.userInfo[NSUnderlyingErrorKey];

        if([underlyingError.domain isEqualToString:_CKUnderlyingErrorDomain] && underlyingError.code == _CKErrorInternalPluginError) {
            NSError* cuttlefishError = underlyingError.userInfo[NSUnderlyingErrorKey];

            if([cuttlefishError.domain isEqualToString:CuttlefishErrorDomain] && cuttlefishError.code == cuttlefishErrorCode) {
                return YES;
            }
        }
    }
    return NO;
}

- (BOOL)_isRetryableNSURLError {
    NSError *error = self;

    if ([error.domain isEqualToString:NSURLErrorDomain]) {
        switch (error.code) {
        case NSURLErrorTimedOut:
        case NSURLErrorNotConnectedToInternet:
            return true;
        default:
            return false;
        }
    }
    return false;
}

- (BOOL)_isRetryableAKError {
    NSError* underlyingError = self.userInfo[NSUnderlyingErrorKey];

    return [self.domain isEqualToString:_AKAppleIDAuthenticationErrorDomain] &&
        underlyingError &&
        [underlyingError _isRetryableNSURLError];
}

- (bool)isRetryable {
    bool retry = false;
    // Specific errors that are transaction failed -- try them again
    if ([self isCuttlefishError:CuttlefishErrorRetryableServerFailure] ||
        [self isCuttlefishError:CuttlefishErrorTransactionalFailure]) {
        retry = true;
    // These are the CuttlefishError -> FunctionErrorType
    } else if ([self isCuttlefishError:CuttlefishErrorJoinFailed] ||
               [self isCuttlefishError:CuttlefishErrorUpdateTrustFailed] ||
               [self isCuttlefishError:CuttlefishErrorEstablishPeerFailed] ||
               [self isCuttlefishError:CuttlefishErrorEstablishBottleFailed] ||
               [self isCuttlefishError:CuttlefishErrorEscrowProxyFailure]) {
        retry = true;
    } else if ([self.domain isEqualToString:TrustedPeersHelperErrorDomain]) {
        switch (self.code) {
        case TrustedPeersHelperErrorCodeUnknownCloudKitError:
            retry = true;
            break;
        default:
            break;
        }
    } else if ([self _isRetryableNSURLError]) {
        retry = true;
    } else if ([self.domain isEqualToString:_CKErrorDomain]) {
        if (self.userInfo[_CKErrorRetryAfterKey] != nil) {
            retry = true;
        } else {
            switch (self.code) {
            case _CKErrorNetworkUnavailable:
            case _CKErrorNetworkFailure:
            case _CKErrorRequestRateLimited:
                retry = true;
                break;
            default:
                break;
            }
        }
    } else if ([self _isCKServerInternalError]) {
        retry = true;
    } else if ([self _isRetryableAKError]) {
        retry = true;
    }

    return retry;
}

static NSTimeInterval _CKRetryAfterSecondsForError(NSError *error) {
    __block NSNumber *lowestRetryAfterSeconds = nil;
    
    if ([error.domain isEqualToString:_CKErrorDomain]) {
        if (error.code != _CKErrorPartialFailure) {
            lowestRetryAfterSeconds = error.userInfo[_CKErrorRetryAfterKey];
        } else {
            NSDictionary<id, NSError *> *partialErrors = error.userInfo[_CKPartialErrorsByItemIDKey];
            [partialErrors enumerateKeysAndObjectsUsingBlock:^(id key, NSError *partialError, BOOL *stop) {
                NSNumber *retryAfterSeconds = partialError.userInfo[_CKErrorRetryAfterKey];
                if (retryAfterSeconds && (lowestRetryAfterSeconds == nil || retryAfterSeconds.doubleValue < lowestRetryAfterSeconds.doubleValue)) {
                    lowestRetryAfterSeconds = retryAfterSeconds;
                }
            }];
        }
    }
    
    if (lowestRetryAfterSeconds != nil) {
        return (NSTimeInterval)lowestRetryAfterSeconds.doubleValue;
    } else {
        return 0;
    }
}

- (NSTimeInterval)cuttlefishRetryAfter {
    NSError *error = self;

    if ([error.domain isEqualToString:_CKErrorDomain] && error.code == _CKErrorServerRejectedRequest) {
        NSError* underlyingError = error.userInfo[NSUnderlyingErrorKey];

        if([underlyingError.domain isEqualToString:_CKUnderlyingErrorDomain] && underlyingError.code == _CKErrorInternalPluginError) {
            NSError* cuttlefishError = underlyingError.userInfo[NSUnderlyingErrorKey];

            if([cuttlefishError.domain isEqualToString:CuttlefishErrorDomain]) {
                NSNumber* val = cuttlefishError.userInfo[CuttlefishErrorRetryAfterKey];
                if (val != nil) {
                    return (NSTimeInterval)val.doubleValue;
                }
            }
        }
    }
    return 0;
}

static NSTimeInterval baseDelay = 30;

+ (void)setDefaultRetryIntervalForTests:(NSTimeInterval)retryInterval {
    baseDelay = retryInterval;
}

- (NSTimeInterval)retryInterval {
    NSTimeInterval ckDelay = _CKRetryAfterSecondsForError(self);
    NSTimeInterval cuttlefishDelay = [self cuttlefishRetryAfter];
    NSTimeInterval delay = MAX(ckDelay, cuttlefishDelay);
    if (delay == 0) {
        delay = baseDelay;
    }
    return delay;
}

@end

#endif // OCTAGON
