/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 15, 2025.
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

#import "keychain/ckks/CloudKitCategories.h"
#import "keychain/ckks/CKKS.h"

@implementation CKOperationGroup (CKKS)
+(instancetype) CKKSGroupWithName:(NSString*)name {
    CKOperationGroup* operationGroup = [[CKOperationGroup alloc] init];
    operationGroup.expectedSendSize = CKOperationGroupTransferSizeKilobytes;
    operationGroup.expectedReceiveSize = CKOperationGroupTransferSizeKilobytes;
    operationGroup.name = name;
    return operationGroup;
}
@end

@implementation NSError (CKKS)

-(bool) ckksIsCKErrorRecordChangedError {
    NSDictionary<CKRecordID*,NSError*>* partialErrors = self.userInfo[CKPartialErrorsByItemIDKey];
    if([self.domain isEqualToString:CKErrorDomain] && self.code == CKErrorPartialFailure && partialErrors) {
        // Check if this error was "you're out of date"

        for(NSError* error in partialErrors.objectEnumerator) {
            if((![error.domain isEqualToString:CKErrorDomain]) || (error.code != CKErrorBatchRequestFailed && error.code != CKErrorServerRecordChanged && error.code != CKErrorUnknownItem)) {
                // There's an error in there that isn't CKErrorServerRecordChanged, CKErrorBatchRequestFailed, or CKErrorUnknownItem. Don't handle nicely...
                return false;
            }
        }

        return true;
    }
    return false;
}

- (BOOL)isCKKSServerPluginError:(NSInteger)code
{
    NSError* underlyingError = self.userInfo[NSUnderlyingErrorKey];
    NSError* thirdLevelError = underlyingError.userInfo[NSUnderlyingErrorKey];

    return ([self.domain isEqualToString:CKErrorDomain] &&
            self.code == CKErrorServerRejectedRequest &&
            underlyingError &&
            [underlyingError.domain isEqualToString:CKUnderlyingErrorDomain] &&
            underlyingError.code == CKUnderlyingErrorPluginError &&
            thirdLevelError &&
            [thirdLevelError.domain isEqualToString:@"CloudkitKeychainService"] &&
            thirdLevelError.code == code);
}

- (BOOL)isCKServerInternalError {
    NSError* underlyingError = self.userInfo[NSUnderlyingErrorKey];

    return [self.domain isEqualToString:CKErrorDomain] &&
        self.code == CKErrorServerRejectedRequest &&
        underlyingError &&
        [underlyingError.domain isEqualToString:CKUnderlyingErrorDomain] &&
        underlyingError.code == CKUnderlyingErrorServerInternalError;
}

- (BOOL)isCKInternalServerHTTPError {
    NSError* underlyingError = self.userInfo[NSUnderlyingErrorKey];

    return [self.domain isEqualToString:CKErrorDomain] &&
        self.code == CKErrorServerRejectedRequest &&
        underlyingError &&
        [underlyingError.domain isEqualToString:CKUnderlyingErrorDomain] &&
        underlyingError.code == CKUnderlyingErrorServerHTTPError;
}
@end

@implementation CKAccountInfo (CKKS)
// Ugly, and might break if CloudKit changes how they print objects. Sorry, CloudKit!
- (NSString*)description {
    NSString* ckprop = [self CKPropertiesDescription];
    NSString* description =  [NSString stringWithFormat: @"<CKAccountInfo: %@>", ckprop];
    return description;
}
@end

#endif //OCTAGON
