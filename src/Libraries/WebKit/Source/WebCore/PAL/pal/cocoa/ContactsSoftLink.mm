/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 17, 2022.
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

#if HAVE(CONTACTS)

#import <Contacts/Contacts.h>
#import <wtf/SoftLinking.h>

SOFT_LINK_FRAMEWORK_FOR_SOURCE_WITH_EXPORT(PAL, Contacts, PAL_EXPORT)
SOFT_LINK_CLASS_FOR_SOURCE_WITH_EXPORT(PAL, Contacts, CNContact, PAL_EXPORT)
SOFT_LINK_CLASS_FOR_SOURCE_WITH_EXPORT(PAL, Contacts, CNLabeledValue, PAL_EXPORT)
SOFT_LINK_CLASS_FOR_SOURCE_WITH_EXPORT(PAL, Contacts, CNPhoneNumber, PAL_EXPORT)
SOFT_LINK_CLASS_FOR_SOURCE_WITH_EXPORT(PAL, Contacts, CNPostalAddress, PAL_EXPORT)
SOFT_LINK_CLASS_FOR_SOURCE_WITH_EXPORT(PAL, Contacts, CNMutableContact, PAL_EXPORT)
SOFT_LINK_CLASS_FOR_SOURCE_WITH_EXPORT(PAL, Contacts, CNMutablePostalAddress, PAL_EXPORT)
SOFT_LINK_CONSTANT_FOR_SOURCE_WITH_EXPORT(PAL, Contacts, CNContactDepartmentNameKey, NSString *, PAL_EXPORT);
SOFT_LINK_CONSTANT_FOR_SOURCE_WITH_EXPORT(PAL, Contacts, CNContactFamilyNameKey, NSString *, PAL_EXPORT);
SOFT_LINK_CONSTANT_FOR_SOURCE_WITH_EXPORT(PAL, Contacts, CNContactGivenNameKey, NSString *, PAL_EXPORT);
SOFT_LINK_CONSTANT_FOR_SOURCE_WITH_EXPORT(PAL, Contacts, CNContactJobTitleKey, NSString *, PAL_EXPORT);
SOFT_LINK_CONSTANT_FOR_SOURCE_WITH_EXPORT(PAL, Contacts, CNContactMiddleNameKey, NSString *, PAL_EXPORT);
SOFT_LINK_CONSTANT_FOR_SOURCE_WITH_EXPORT(PAL, Contacts, CNContactNamePrefixKey, NSString *, PAL_EXPORT);
SOFT_LINK_CONSTANT_FOR_SOURCE_WITH_EXPORT(PAL, Contacts, CNContactNameSuffixKey, NSString *, PAL_EXPORT);
SOFT_LINK_CONSTANT_FOR_SOURCE_WITH_EXPORT(PAL, Contacts, CNContactNicknameKey, NSString *, PAL_EXPORT);
SOFT_LINK_CONSTANT_FOR_SOURCE_WITH_EXPORT(PAL, Contacts, CNContactNoteKey, NSString *, PAL_EXPORT);
SOFT_LINK_CONSTANT_FOR_SOURCE_WITH_EXPORT(PAL, Contacts, CNContactOrganizationNameKey, NSString *, PAL_EXPORT);
SOFT_LINK_CONSTANT_FOR_SOURCE_WITH_EXPORT(PAL, Contacts, CNContactPhoneticFamilyNameKey, NSString *, PAL_EXPORT);
SOFT_LINK_CONSTANT_FOR_SOURCE_WITH_EXPORT(PAL, Contacts, CNContactPhoneticGivenNameKey, NSString *, PAL_EXPORT);
SOFT_LINK_CONSTANT_FOR_SOURCE_WITH_EXPORT(PAL, Contacts, CNContactPhoneticMiddleNameKey, NSString *, PAL_EXPORT);
SOFT_LINK_CONSTANT_FOR_SOURCE_WITH_EXPORT(PAL, Contacts, CNContactPhoneticOrganizationNameKey, NSString *, PAL_EXPORT);
SOFT_LINK_CONSTANT_FOR_SOURCE_WITH_EXPORT(PAL, Contacts, CNContactPreviousFamilyNameKey, NSString *, PAL_EXPORT);
SOFT_LINK_CONSTANT_FOR_SOURCE_WITH_EXPORT(PAL, Contacts, CNContactBirthdayKey, NSString *, PAL_EXPORT);
SOFT_LINK_CONSTANT_FOR_SOURCE_WITH_EXPORT(PAL, Contacts, CNContactNonGregorianBirthdayKey, NSString *, PAL_EXPORT);
SOFT_LINK_CONSTANT_FOR_SOURCE_WITH_EXPORT(PAL, Contacts, CNContactPhoneNumbersKey, NSString *, PAL_EXPORT);
SOFT_LINK_CONSTANT_FOR_SOURCE_WITH_EXPORT(PAL, Contacts, CNContactEmailAddressesKey, NSString *, PAL_EXPORT);
SOFT_LINK_CONSTANT_FOR_SOURCE_WITH_EXPORT(PAL, Contacts, CNContactPostalAddressesKey, NSString *, PAL_EXPORT);
SOFT_LINK_CONSTANT_FOR_SOURCE_WITH_EXPORT(PAL, Contacts, CNContactDatesKey, NSString *, PAL_EXPORT);
SOFT_LINK_CONSTANT_FOR_SOURCE_WITH_EXPORT(PAL, Contacts, CNContactUrlAddressesKey, NSString *, PAL_EXPORT);

#endif // HAVE(CONTACTS)
