/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 18, 2025.
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
#import "config.h"
#import "SerializedPlatformDataCueValue.h"

#import <AVFoundation/AVMetadataItem.h>
#import <wtf/cocoa/TypeCastsCocoa.h>
#import <wtf/text/WTFString.h>

namespace WebCore {

SerializedPlatformDataCueValue::SerializedPlatformDataCueValue(AVMetadataItem *item)
{
    if (!item)
        return;

    m_data = Data { };

    auto dictionary = adoptNS([[NSMutableDictionary alloc] init]);
    NSDictionary *extras = [item extraAttributes];

    for (id key in [extras keyEnumerator]) {
        if (![key isKindOfClass:[NSString class]])
            continue;
        NSString *value = [extras objectForKey:key];
        if (![value isKindOfClass:NSString.class])
            continue;
        NSString *keyString = key;

        if ([key isEqualToString:@"MIMEtype"])
            keyString = @"type";
        else if ([key isEqualToString:@"dataTypeNamespace"] || [key isEqualToString:@"pictureType"])
            continue;
        else if ([key isEqualToString:@"dataType"]) {
            id dataTypeNamespace = [extras objectForKey:@"dataTypeNamespace"];
            if (!dataTypeNamespace || ![dataTypeNamespace isKindOfClass:[NSString class]] || ![dataTypeNamespace isEqualToString:@"org.iana.media-type"])
                continue;
            keyString = @"type";
        } else {
            if (![value length])
                continue;
            keyString = [key lowercaseString];
        }

        if ([keyString isEqualToString:@"type"])
            m_data->type = value;
        else
            m_data->otherAttributes.add(keyString, value);
    }

    if ([item.key isKindOfClass:NSString.class])
        m_data->key = (NSString *)item.key;

    if (item.locale)
        m_data->locale = item.locale;

    if (auto *str = dynamic_objc_cast<NSString>(item.value))
        m_data->value = str;
    else if (auto *data = dynamic_objc_cast<NSData>(item.value))
        m_data->value = data;
    else if (auto *date = dynamic_objc_cast<NSDate>(item.value))
        m_data->value = date;
    else if (auto *number = dynamic_objc_cast<NSNumber>(item.value))
        m_data->value = number;
}

RetainPtr<NSDictionary> SerializedPlatformDataCueValue::toNSDictionary() const
{
    if (!m_data)
        return nullptr;

    auto dictionary = adoptNS([NSMutableDictionary new]);

    if (!m_data->type.isNull())
        [dictionary setObject:m_data->type forKey:@"type"];

    for (auto& pair : m_data->otherAttributes)
        [dictionary setObject:pair.value forKey:pair.key];

    if (m_data->locale)
        [dictionary setObject:m_data->locale.get() forKey:@"locale"];

    if (!m_data->key.isNull())
        [dictionary setObject:m_data->key forKey:@"key"];

    WTF::switchOn(m_data->value, [] (std::nullptr_t) {
    }, [&] (RetainPtr<NSString> string) {
        [dictionary setValue:string.get() forKey:@"data"];
    }, [&] (RetainPtr<NSNumber> number) {
        [dictionary setValue:number.get() forKey:@"data"];
    }, [&] (RetainPtr<NSData> data) {
        [dictionary setValue:data.get() forKey:@"data"];
    }, [&] (RetainPtr<NSDate> date) {
        [dictionary setValue:date.get() forKey:@"data"];
    });

    return dictionary;
}

bool SerializedPlatformDataCueValue::operator==(const SerializedPlatformDataCueValue& other) const
{
    if (!m_data || !other.m_data)
        return false;

    return *m_data == *other.m_data;
}

bool SerializedPlatformDataCueValue::Data::operator==(const Data& other) const
{
    return type == other.type
        && otherAttributes == other.otherAttributes
        && [locale isEqual:other.locale.get()]
        && key == other.key
        && value.index() == other.value.index()
        && WTF::switchOn(value, [] (std::nullptr_t) {
            return true;
        }, [&] (RetainPtr<NSString> string) {
            return !![string isEqual:std::get<RetainPtr<NSString>>(other.value).get()];
        }, [&] (RetainPtr<NSNumber> number) {
            return !![number isEqual:std::get<RetainPtr<NSNumber>>(other.value).get()];
        }, [&] (RetainPtr<NSData> data) {
            return !![data isEqual:std::get<RetainPtr<NSData>>(other.value).get()];
        }, [&] (RetainPtr<NSDate> date) {
            return !![date isEqual:std::get<RetainPtr<NSDate>>(other.value).get()];
        });
}

}
