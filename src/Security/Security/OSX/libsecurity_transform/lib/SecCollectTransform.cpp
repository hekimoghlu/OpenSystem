/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 8, 2023.
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
#include "Transform.h"
#include "SecTransform.h"
#include "SecCollectTransform.h"
#include "SecCustomTransform.h"
#include "misc.h"
#include "c++utils.h"
#include "Utilities.h"

static CFStringRef kCollectTransformName = CFSTR("com.apple.security.seccollecttransform");

static SecTransformInstanceBlock CollectTransform(CFStringRef name, 
							SecTransformRef newTransform, 
							SecTransformImplementationRef ref)
{
	SecTransformInstanceBlock instanceBlock = 
	^{
		__block CFMutableArrayRef allValues = 
			CFArrayCreateMutable(kCFAllocatorDefault, 0, &kCFTypeArrayCallBacks);
		__block Boolean isSameType = TRUE;
		CFTypeRef input_ah = SecTranformCustomGetAttribute(ref, kSecTransformInputAttributeName, kSecTransformMetaAttributeRef);
		ah2ta(input_ah)->direct_error_handling = 1;
		
		dispatch_block_t no_more_output = ^
		{
			SecTransformSetAttributeAction(ref, kSecTransformActionAttributeNotification, input_ah, ^(SecTransformStringOrAttributeRef a, CFTypeRef v) { return v; });
		};
		
		// Create a block to deal with out of memory errors
		dispatch_block_t oom = ^ 
		{
			CFTypeRefHolder localErr(GetNoMemoryError());
			SecTransformCustomSetAttribute(ref, kSecTransformOutputAttributeName, 
					kSecTransformMetaAttributeValue, localErr.Get());
			no_more_output();
		};
		
		SecTransformSetTransformAction(ref, kSecTransformActionFinalize,
			^()
				{
					if (NULL != allValues)
					{
						CFReleaseNull(allValues);
					}
					
					return (CFTypeRef) NULL;
				});

		SecTransformSetAttributeAction(ref, kSecTransformActionAttributeNotification,
			input_ah, 
			^(SecTransformStringOrAttributeRef attribute, CFTypeRef value)
			{
				CFIndex len = CFArrayGetCount(allValues);

#if 0
				if (NULL == value && 0 == len)
				{
					SecTransformCustomSetAttribute(ref, kSecTransformOutputAttributeName,
					kSecTransformMetaAttributeValue, NULL);
					no_more_output();
					return value;
				}
#endif
				
				if (value && isSameType && len > 0)
				{
					isSameType = CFGetTypeID(CFArrayGetValueAtIndex(allValues, 0)) == CFGetTypeID(value);
				}

				if (value) 
				{

					if (CFGetTypeID(value) == CFErrorGetTypeID()) 
					{
						SecTransformCustomSetAttribute(ref, kSecTransformOutputAttributeName, 
								kSecTransformMetaAttributeValue, value);
						no_more_output();
						return value;
					}

					// For mutable types, we want an immutable copy.   
					/// XXX: write a more general CFImutableCopy and use it here.
					if (CFGetTypeID(value) == CFDataGetTypeID()) 
					{
						CFDataRef copy = CFDataCreateCopy(NULL, (CFDataRef)value);
						
						CFArrayAppendValue(allValues, copy);
						CFReleaseNull(copy);
					} 
					else 
					{
						CFArrayAppendValue(allValues, value);
					}

					if (CFArrayGetCount(allValues) != len +1) 
					{
						oom();									
						return value;
					}
				}
				else
				{
					if (isSameType) 
					{
						// Deal with data or no items at all
						CFTypeID type = CFArrayGetCount(allValues) ? 
							CFGetTypeID(CFArrayGetValueAtIndex(allValues, 0)) : CFDataGetTypeID();
						if (CFDataGetTypeID() == type) 
						{
							CFIndex total_len = 0;
							CFIndex prev_total_len = 0;
							CFIndex i;
							const CFIndex n_datas = CFArrayGetCount(allValues);

							for(i = 0; i < n_datas; i++) 
							{
								total_len += 
									CFDataGetLength((CFDataRef)CFArrayGetValueAtIndex(allValues, i));
								if (total_len < prev_total_len) 
								{
									oom();
									return value;
								}
								prev_total_len = total_len;
							}

							CFMutableDataRef result = CFDataCreateMutable(NULL, total_len);
							if (!result) 
							{
								oom();
								return value;
							}

							for(i = 0; i < n_datas; i++) 
							{
								CFDataRef d = (CFDataRef)CFArrayGetValueAtIndex(allValues, i);
								CFDataAppendBytes(result, CFDataGetBytePtr(d), CFDataGetLength(d));
							}

							if (CFDataGetLength(result) != total_len) 
							{
								oom();
                                CFReleaseNull(result);
								return value;
							}

							CFDataRef resultData = CFDataCreateCopy(NULL, result);
							
							SecTransformCustomSetAttribute(ref, kSecTransformOutputAttributeName, 
									kSecTransformMetaAttributeValue, (CFTypeRef)resultData);
									
							CFReleaseNull(resultData);
							
							SecTransformCustomSetAttribute(ref, kSecTransformOutputAttributeName, 
									kSecTransformMetaAttributeValue, (CFTypeRef)value);
							no_more_output();

							CFReleaseNull(result);
							return value;
						} 
						else if (CFStringGetTypeID() == type) 
						{
							// deal with strings
							CFStringRef resultStr = CFStringCreateByCombiningStrings(NULL, allValues, CFSTR(""));
							
							SecTransformCustomSetAttribute(ref, kSecTransformOutputAttributeName, 
									kSecTransformMetaAttributeValue, (CFTypeRef)resultStr);
							SecTransformCustomSetAttribute(ref, kSecTransformOutputAttributeName, 
									kSecTransformMetaAttributeValue, (CFTypeRef)value);
							no_more_output();
                            CFReleaseNull(resultStr);
		
							return value;
						} 
						else 
						{
							// special case the singleton
							if (1 == CFArrayGetCount(allValues))
							{
								CFTypeRef result =  (CFTypeRef)CFRetainSafe(CFArrayGetValueAtIndex(allValues, 0));
								
								SecTransformCustomSetAttribute(ref, kSecTransformOutputAttributeName, 
										kSecTransformMetaAttributeValue, (CFTypeRef)result);
								SecTransformCustomSetAttribute(ref, kSecTransformOutputAttributeName, 
										kSecTransformMetaAttributeValue, (CFTypeRef)value);
								no_more_output();
                                CFReleaseNull(result);

								return value;
							}
						}
					}
					// Fall through for non-homogenous or un-mergable type
					CFArrayRef resultArray = CFArrayCreateCopy(kCFAllocatorDefault, allValues);
					SecTransformCustomSetAttribute(ref, kSecTransformOutputAttributeName, 
							kSecTransformMetaAttributeValue, (CFTypeRef)resultArray);
					SecTransformCustomSetAttribute(ref, kSecTransformOutputAttributeName, 
							kSecTransformMetaAttributeValue, (CFTypeRef)value);
					no_more_output();
                    CFReleaseNull(resultArray);

					return value;
				}

				return value;

			});
										
		return (CFErrorRef)NULL;
	};
	
	return Block_copy(instanceBlock);	
}

SecTransformRef SecCreateCollectTransform(CFErrorRef* error) 
{
	static dispatch_once_t once;
	__block Boolean ok = TRUE;
			
	dispatch_block_t aBlock = ^
	{
		ok = SecTransformRegister(kCollectTransformName, &CollectTransform, error);
	};
	
	dispatch_once(&once, aBlock);

	if (!ok) 
	{
		return NULL;
	}

	SecTransformRef yatz = SecTransformCreate(kCollectTransformName, error);
	return yatz;
}
