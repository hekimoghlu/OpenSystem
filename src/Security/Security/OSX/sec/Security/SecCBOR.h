/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 17, 2023.
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
   SecCBOR.h
*/

#import <Foundation/Foundation.h>
#import <Security/Security.h>

NS_ASSUME_NONNULL_BEGIN

// NOTE: This is not a full CBOR implementation, only the writer has been implemented
// along with the types below implemented.  The implemented field types are CTAP2 compliant
//
// https://w3c.github.io/webauthn/#sctn-conforming-all-classes
// https://fidoalliance.org/specs/fido-v2.0-ps-20190130/fido-client-to-authenticator-protocol-v2.0-ps-20190130.html#ctap2-canonical-cbor-encoding-form
typedef enum {
    SecCBORType_Unsigned = 0,
    SecCBORType_Negative = 1,
    SecCBORType_ByteString = 2,
    SecCBORType_String = 3,
    SecCBORType_Array = 4,
    SecCBORType_Map = 5,
    SecCBORType_None = -1,
} SecCBORType;


NS_REQUIRES_PROPERTY_DEFINITIONS @interface SecCBORValue: NSObject

@property (nonatomic, readonly) SecCBORType fieldType;
@property (nonatomic, readonly) uint8_t fieldValue;

- (void)write:(NSMutableData *)output;
- (void)encodeStartItems:(uint64_t)items output:(NSMutableData *)output;
- (void)setAdditionalInformation:(uint8_t)item1 item2:(uint8_t)additionalInformation output:(NSMutableData *)output;
- (void)setUint:(uint8_t)item1 item2:(uint64_t)value output:(NSMutableData *)output;

@end


NS_REQUIRES_PROPERTY_DEFINITIONS @interface SecCBORUnsigned : SecCBORValue {
    NSUInteger m_data;
}
- (instancetype)initWith:(NSUInteger)data;
- (void)write:(NSMutableData *)output;
- (NSComparisonResult)compare:(SecCBORUnsigned *)target;
- (NSString *)getLabel;

@end


NS_REQUIRES_PROPERTY_DEFINITIONS @interface SecCBORNegative : SecCBORValue {
    NSInteger m_data;
}
- (instancetype)initWith:(NSInteger)data;
- (void)write:(NSMutableData *)output;
- (NSComparisonResult)compare:(SecCBORNegative *)target;
- (NSString *)getLabel;

@end


NS_REQUIRES_PROPERTY_DEFINITIONS @interface SecCBORString : SecCBORValue {
    NSString *m_data;
}
- (instancetype)initWith:(NSString *)data;
- (void)write:(NSMutableData *)output;
- (NSComparisonResult)compare:(SecCBORString *)target;
- (NSString *)getLabel;

@end


NS_REQUIRES_PROPERTY_DEFINITIONS @interface SecCBORData: SecCBORValue {
    NSData *m_data;
}
- (instancetype)initWith:(NSData *)data;
- (void)write:(NSMutableData *)output;

@end


NS_REQUIRES_PROPERTY_DEFINITIONS @interface SecCBORArray: SecCBORValue {
    NSMutableArray *m_data;
}
- (instancetype)init;
- (instancetype)initWith:(NSArray *)data;
- (void)addObject:(SecCBORValue *)object;
- (void)write:(NSMutableData *)output;

@end


NS_REQUIRES_PROPERTY_DEFINITIONS @interface SecCBORMap: SecCBORValue {
    NSMapTable *m_data;
}
- (instancetype)init;
- (void)setKey:(SecCBORValue *)key value:(SecCBORValue *)data;
- (NSArray *)getSortedKeys;
- (NSDictionary *)dictionaryRepresentation;
- (void)write:(NSMutableData *)output;

@end

NS_ASSUME_NONNULL_END
