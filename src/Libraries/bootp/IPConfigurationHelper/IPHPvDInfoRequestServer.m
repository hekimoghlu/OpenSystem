/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 14, 2022.
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
// Reference: [RFC8801](https://datatracker.ietf.org/doc/html/rfc8801)

#import <Foundation/NSCalendarDate.h>
#import <Foundation/NSURLConnectionPrivate.h>

#import "cfutil.h"
#import "DNSNameList.h"
#import "IPConfigurationLog.h"
#import "IPConfigurationPrivate.h"
#import "IPHPvDInfoRequest.h"
#import "IPHPvDInfoRequestUtil.h"

#define kPvDInfoURLSchema			"https://"
#define kPvDInfoURLResourceExtension		"/.well-known/pvd"
#define kPvDInfoHTTPMethod			"GET"
#define kPvDInfoHTTPHeaderContentTypeKey	"Content-Type"
#define kPvDInfoHTTPHeaderContentTypeVal	"application/pvd+json"
#define kPvDInfoHTTPHeaderAcceptKey		"Accept"
#define kPvDInfoHTTPHeaderAcceptVal 		kPvDInfoHTTPHeaderContentTypeVal

#define COLLECTION_TYPE_FIELD_MAX_COUNT 	10
#define NESTING_LEVEL_MAX 			2
#define ELEMENTS_PER_LEVEL_MAX 			COLLECTION_TYPE_FIELD_MAX_COUNT
#define NECESSARY_FIELDS_COUNT 			3
#define NECESSARY_FIELD_IDENTIFIER 		kPvDInfoAdditionalInfoDictKeyIdentifierCStr
#define NECESSARY_FIELD_EXPIRES 		kPvDInfoAdditionalInfoDictKeyExpiresCStr
#define NECESSARY_FIELD_PREFIXES 		kPvDInfoAdditionalInfoDictKeyPrefixesCStr
#define OPTIONAL_FIELDS_COUNT 			2
#define OPTIONAL_FIELD_DNS_ZONES 		"dnsZones"
#define OPTIONAL_FIELD_NO_INTERNET 		"noInternet"
#define EXTRA_FIELDS_COUNT 			1
#define EXTRA_FIELD_PROXIES 			"proxies"

@interface IPHPvDInfoRequestServer ()
@property (nonatomic, readwrite) NSURLSession * urlSession;
@property (nonatomic, readwrite) BOOL validFetch;
@property (nonatomic, copy) void (^xpcClientCompletionHandler)(NSDictionary *);
@end

@implementation IPHPvDInfoRequestServer

@synthesize urlSession = _urlSession;
@synthesize validFetch = _validFetch;
@synthesize xpcClientCompletionHandler = _xpcClientCompletionHandler;

- (instancetype)init
{
	self = [super init];
	
	if (self) {
		_IPConfigurationInitLog(kIPConfigurationLogCategoryHelper);
		self.validFetch = YES;
	}
	
	return self;
}

/* 
 * Skips scheduled invocations of 'xpcClientCompletionHandler()'.
 */
- (void)cancelRequest {
	if (self.urlSession != nil) {
		IPConfigLog(LOG_INFO, "cancelling in-flight URLSession and tasks");
		[self.urlSession invalidateAndCancel];
		self.urlSession = nil;
	}
}

/*
 * Implements protocol NSURLSessionTaskDelegate
 */
- (void)URLSession:(NSURLSession *)session
	      task:(NSURLSessionTask *)task
didReceiveChallenge:(NSURLAuthenticationChallenge *)challenge
 completionHandler:(void (^)(NSURLSessionAuthChallengeDisposition disposition,
			     NSURLCredential * credential))completionHandler
{
	IPConfigLog(LOG_DEBUG, "entered authentication challenge callback");
#ifdef __TEST_IPH_PVD__
	// Note: Build this with 'buildit ... -othercflags "-D__TEST_IPH_PVD__"' for testing
	if ([[[challenge protectionSpace] authenticationMethod]
	     isEqualToString:NSURLAuthenticationMethodServerTrust]) {
		IPConfigLog(LOG_NOTICE, "encountered a server authentication challenge");
		/*
		 * Overrides default secure transports TLS certificate chain
		 * validation, so it allows connecting to a server that
		 * presents a self-signed certificate.
		 */
		IPConfigLog(LOG_NOTICE, "trusting server certificate...");
		completionHandler(NSURLSessionAuthChallengeUseCredential,
				  [NSURLCredential credentialForTrust:challenge.protectionSpace.serverTrust]);
		goto done;
	}
#endif // __TEST_IPH_PVD__
	/*
	 * Follows the default behavior from Secure Transports.
	 */
	completionHandler(NSURLSessionAuthChallengePerformDefaultHandling, nil);

#ifdef __TEST_IPH_PVD__
done:
#endif // __TEST_IPH_PVD
	IPConfigLog(LOG_DEBUG, "finished authentication challenge callback");
	return;
}

/*
 * Implements protocol IPConfigurationHelperPvDInfoProtocol
 */
- (void)fetchPvDAdditionalInformationWithPvDID:(NSString *)pvdID
				 prefixesArray:(NSArray<NSString *> *)prefixes
			       bindToInterface:(NSString *)ifName
			  andCompletionHandler:(void (^)(NSDictionary *))completion
{
	NSString *urlString = nil;
	NSURL *fetchURL = nil;
	NSMutableURLRequest *httpRequest = nil;
	NSURLSessionDataTask *dataTask = nil;
	
	/* saves the completion block for later */
	self.xpcClientCompletionHandler = completion;
	
	/*
	 * The ephemeral NSURLSession doesn't store any HTTP cache or cookies.
	 * Will delegate authorization challenges to this IPConfigurationHelper.
	 */
	self.urlSession = [NSURLSession sessionWithConfiguration:[NSURLSessionConfiguration ephemeralSessionConfiguration]
							delegate:self
						   delegateQueue:nil];
	if (self.urlSession == nil) {
		IPConfigLog(LOG_ERR, "failed to create an NSURLSession");
		return;
	}
	/*
	 * The URL root string is expected to be just the PvD-ID, per RFC 8801.
	 * The fully qualified URL looks like 'https://<PvD-ID>/.well-known/pvd'
	 */
	urlString = [NSString stringWithFormat:@"%s%@%s", kPvDInfoURLSchema, pvdID, kPvDInfoURLResourceExtension];
	fetchURL = [NSURL URLWithString:urlString];
	/*
	 * HTTP request format for PvD Additional Info:
	 * Method: GET
	 * Content-Type: application/pvd+json
	 * Accept: application/pvd+json
	 */
	httpRequest = [[NSMutableURLRequest alloc] initWithURL:fetchURL];
	[httpRequest setHTTPMethod:@kPvDInfoHTTPMethod];
	[httpRequest setValue:@kPvDInfoHTTPHeaderContentTypeVal
	   forHTTPHeaderField:@kPvDInfoHTTPHeaderContentTypeKey];
	[httpRequest setValue:@kPvDInfoHTTPHeaderAcceptVal
	   forHTTPHeaderField:@kPvDInfoHTTPHeaderAcceptKey];
	[httpRequest setBoundInterfaceIdentifier:ifName];
	dataTask = [self.urlSession dataTaskWithRequest:(NSURLRequest *)httpRequest
				      completionHandler:^(NSData * _Nullable data,
							  NSURLResponse * _Nullable response,
							  NSError * _Nullable error) {
		id retrievedObject = nil;
		NSDictionary *parsedJSONDict = nil;
		NSError *jsonError = nil;
		NSHTTPURLResponse *httpResponse = nil;
		long statusCode = -1;

		if (error != nil) {
			IPConfigLog(LOG_ERR, "failed NSURLSessionDataTask with error '%@'", error);
			if ([error.domain isEqualToString:(__bridge NSString *)kCFErrorDomainCFNetwork]
			    && error.code == kCFURLErrorNotConnectedToInternet) {
				/* OK to retry this later with regained connectivity */
				IPConfigLog(LOG_NOTICE, "no internet connection currently");
			} else {
				/* fail anything that's not "no internet right now" */
				self.validFetch = NO;
			}
			goto done;
		}
		/*
		 * HTTP status codes for PvD Additional Info:
		 * 200-299 : single JSON object in retrieved data
		 * 300-399 : follow redirections
		 * >= 400 : no PvD additional info
		 */
		if (response == nil) {
			IPConfigLog(LOG_ERR, "got NULL NSURLResponse");
			self.validFetch = NO;
			goto done;
		}
		httpResponse = (NSHTTPURLResponse *)response;
		statusCode = (long)[httpResponse statusCode];
		/* validates retrieved HTTP response */
		if (statusCode >= 200 && statusCode <= 299) {
			/* SUCCESS */
		} else if (statusCode >= 300 && statusCode <= 399) {
			/* HTTP redirects are followed automatically by this dataTask */
		} else if (statusCode >= 400) {
			IPConfigLog(LOG_ERR, "encountered HTTP error '%ld' (%@)", statusCode,
				[NSHTTPURLResponse localizedStringForStatusCode:statusCode]);
			self.validFetch = NO;
			goto done;
		} else {
			IPConfigLog(LOG_ERR, "unrecognized HTTP status code '%ld'", statusCode);
			self.validFetch = NO;
			goto done;
		}
		/* validates retrieved JSON data */
		if (data == nil) {
			IPConfigLog(LOG_ERR, "retrieved NULL data from URL '%@'", urlString);
			self.validFetch = NO;
			goto done;
		}
		retrievedObject = [NSJSONSerialization JSONObjectWithData:data
								  options:NSJSONReadingTopLevelDictionaryAssumed
								    error:&jsonError];
		if (jsonError != nil || retrievedObject == nil) {
			IPConfigLog(LOG_ERR, "failed JSON parsing with error '%@'", jsonError);
			self.validFetch = NO;
			goto done;
		}
		if (![retrievedObject isKindOfClass:[NSDictionary class]]) {
			IPConfigLog(LOG_ERR, "parsed JSON object isn't a well-formed NSDictionary");
			self.validFetch = NO;
			goto done;
		}
		parsedJSONDict = (NSDictionary *)retrievedObject;
		IPConfigLog(LOG_INFO, "fetched PvD Additional Info JSON object:\n'%@'", parsedJSONDict);
		
	done:
		if (self.validFetch && parsedJSONDict != nil) {
			[self scheduleParsingEventCompleteWithParsedJSON:parsedJSONDict
								   pvdID:pvdID
							    ipv6Prefixes:prefixes];
		} else {
			[self scheduleParsingEventAbort];
		}
		return;
	}];
	
	IPConfigLog(LOG_INFO, "fetching PvD Additional Info from URL %@", urlString);
	[dataTask resume];
	return;
}

- (void)scheduleParsingEventCompleteWithParsedJSON:(NSDictionary *)dictionary
					     pvdID:(NSString *)pvdid
				      ipv6Prefixes:(NSArray<NSString *> *)prefixes
{
	dispatch_async(dispatch_get_main_queue(), ^{
		CFMutableDictionaryRef xpcRetDict = NULL;
		CFDictionaryRef additionalInfoDict = NULL;
		
		if (self.urlSession == nil) {
			/* PvDInfoRequest has been invalidated */
			goto done;
		}
		IPConfigLog(LOG_DEBUG, "url fetch completed successfully");
		xpcRetDict = CFDictionaryCreateMutable(NULL, 2,
						       &kCFTypeDictionaryKeyCallBacks,
						       &kCFTypeDictionaryValueCallBacks);
		additionalInfoDict = [self createValidPvDAdditionalInfoDict:dictionary withID:pvdid andPrefixes:prefixes];
		if (additionalInfoDict == NULL) {
			self.validFetch = NO;
		}
		CFDictionarySetValue(xpcRetDict, kPvDInfoValidFetchXPCKey,
				     self.validFetch ? kCFBooleanTrue : kCFBooleanFalse);
		if (self.validFetch && additionalInfoDict != NULL) {
			IPConfigLog(LOG_DEBUG, "fetched pvd info was validated");
			CFDictionarySetValue(xpcRetDict, kPvDInfoAdditionalInfoDictXPCKey, additionalInfoDict);
		}
		[self.urlSession finishTasksAndInvalidate];
		self.urlSession = nil;
		self.xpcClientCompletionHandler((__bridge_transfer NSDictionary *)xpcRetDict);
		
	done:
		my_CFRelease(&additionalInfoDict);
		return;
	});
	return;
}

/*
 * For the case of no current internet connectivity, this returns a valid fetch
 * with a nil addinfo dictionary. Any other failures are considered fatal.
 */
- (void)scheduleParsingEventAbort
{
	dispatch_async(dispatch_get_main_queue(), ^{
		CFMutableDictionaryRef xpcRetDict = NULL;

		if (self.urlSession == nil) {
			/* PvDInfoRequest has been invalidated */
			goto done;
		}
		IPConfigLog(LOG_NOTICE, "aborting parsing event due to a failed url fetch");
		[self.urlSession invalidateAndCancel];
		self.urlSession = nil;
		xpcRetDict = CFDictionaryCreateMutable(NULL, 1,
						       &kCFTypeDictionaryKeyCallBacks,
						       &kCFTypeDictionaryValueCallBacks);
		CFDictionarySetValue(xpcRetDict, kPvDInfoValidFetchXPCKey,
				     self.validFetch ? kCFBooleanTrue : kCFBooleanFalse);
		self.xpcClientCompletionHandler((__bridge_transfer NSDictionary *)xpcRetDict);

	done:
		return;
	});
	return;
}

static bool
copy_valid_identifier(CFStringRef * identifier, NSString * field, id val, NSString * pvdID)
{
	/* Must be String<FQDN> */
	CFDataRef isValidFQDN = NULL;
	bool success = false;

	IPConfigLog(LOG_DEBUG, "validating field '%@' with value '%@' of class '%@'", field, val, [val class]);
	if (![val isKindOfClass:[NSString class]]) {
		IPConfigLog(LOG_NOTICE, "expected String element, got '%@'", [val class]);
		goto done;
	}
	if (![[val lowercaseString] isEqualToString:[pvdID lowercaseString]]) {
		IPConfigLog(LOG_NOTICE, "retrieved ID must be an equal string with RA's PvD ID");
		goto done;
	}
	if ((isValidFQDN = DNSNameListDataCreateWithString((__bridge CFStringRef)val)) == NULL) {
		IPConfigLog(LOG_NOTICE, "couldn't validate PvD '%@' as an FQDN", field);
		goto done;
	}
	CFRelease(isValidFQDN);
	*identifier = CFStringCreateCopy(NULL, (__bridge CFStringRef)val);
	if (*identifier == NULL) {
		IPConfigLog(LOG_NOTICE, "couldn't copy id string");
		goto done;
	}
	success = true;

done:
	return success;
}

static bool
copy_valid_expires(CFStringRef * expires, NSString * field, id val)
{
	/* Must be Date */
	CFDateFormatterRef dateFormatter = NULL;
	CFLocaleRef locale = NULL;
	CFDateFormatterStyle style;
	CFDateRef date = NULL;
	CFDateRef nowDate = NULL;
	bool success = false;

	IPConfigLog(LOG_DEBUG, "validating field '%@' with value '%@' of class '%@'", field, val, [val class]);
	if (![val isKindOfClass:[NSString class]]) {
		IPConfigLog(LOG_NOTICE, "expected String element, got '%@'", [val class]);
		goto done;
	}
	style = kCFDateFormatterNoStyle;
	locale = CFLocaleCreate(NULL, kPvDInfoExpirationDateLocale);
	if (locale == NULL) {
		goto done;
	}
	dateFormatter = CFDateFormatterCreate(NULL, locale, style, style);
	if (dateFormatter == NULL) {
		goto done;
	}
	CFDateFormatterSetFormat(dateFormatter, kPvDInfoExpirationDateFormat);
	date = CFDateFormatterCreateDateFromString(NULL,
						   dateFormatter,
						   (__bridge CFStringRef)val,
						   NULL);
	if (date == NULL) {
		goto done;
	}
#ifndef __TEST_IPH_PVD__
	nowDate = CFDateCreate(NULL, CFAbsoluteTimeGetCurrent());
	if (nowDate == NULL) {
		goto done;
	}
	/* retrieved expiration time must be greater than current time */
	if (CFDateCompare(date, nowDate, NULL) != kCFCompareGreaterThan) {
		IPConfigLog(LOG_NOTICE, "expiration date must be in the future");
		goto done;
	}
#endif // __TEST_IPH_PVD__
	*expires = CFDateFormatterCreateStringWithDate(NULL, dateFormatter, date);
	if (*expires == NULL) {
		IPConfigLog(LOG_NOTICE, "failed to create string from date formatter");
		goto done;
	}
	success = true;

done:
	my_CFRelease(&locale);
	my_CFRelease(&dateFormatter);
	my_CFRelease(&date);
	my_CFRelease(&nowDate);
	return success;
}

static bool
copy_valid_prefixes(CFArrayRef * prefixes, NSString * field, id val, NSArray<NSString *> * raPIOPrefixes)
{
	/* Must be Array<String> */
	/* String must be in CIDR representation "IPv6_ADDR/PREFIXLEN" */
	CFMutableArrayRef prefixStringsCIDR = NULL;
	NSMutableArray<NSString *> *prefixStrings = NULL;
	NSUInteger arrayCount = [(NSArray *)val count];
	NSUInteger arrayCountMax = ((arrayCount < COLLECTION_TYPE_FIELD_MAX_COUNT)
				    ? arrayCount
				    : COLLECTION_TYPE_FIELD_MAX_COUNT);
	CFArrayRef splitter = NULL;
	bool wellFormed = true;
	bool contained = false;
	bool success = false;

	IPConfigLog(LOG_DEBUG, "validating field '%@' with value '%@' of class '%@'", field, val, [val class]);
	if (![val isKindOfClass:[NSArray class]]) {
		IPConfigLog(LOG_NOTICE, "expected JSON value of Array type");
		goto done;
	}
	prefixStrings = [NSMutableArray array];
	prefixStringsCIDR = CFArrayCreateMutable(NULL, 0, &kCFTypeArrayCallBacks);
	if (prefixStrings == nil || prefixStringsCIDR == NULL) {
		goto done;
	}
	for (NSUInteger i = 0;  i < arrayCountMax; i++) {
		@autoreleasepool {
			CFStringRef prefixStr = NULL;
			const void *prefixAddrPtr = NULL;
			CFStringRef prefixAddrStr = NULL;
			struct in6_addr prefixAddr = {0};
			const void *prefixLenPtr = NULL;
			CFStringRef prefixLenStr = NULL;
			uint32_t prefixLenUInt = 0;
			CFStringRef copyStr = NULL;

			prefixStr = (__bridge CFStringRef)[(NSArray *)val objectAtIndex:i];
			if (!isA_CFString(prefixStr)) {
				IPConfigLog(LOG_NOTICE, "expected '%@' element, got '%@'",
					    CFCopyTypeIDDescription(CFStringGetTypeID()),
					    CFCopyTypeIDDescription(CFGetTypeID(prefixStr)));
				wellFormed = false;
				break;
			}
			splitter = CFStringCreateArrayBySeparatingStrings(NULL, prefixStr, CFSTR("/"));
			if (splitter == NULL || CFArrayGetCount(splitter) != 2) {
				IPConfigLog(LOG_NOTICE, "couldn't split provided string");
				wellFormed = false;
				break;
			}
			prefixAddrPtr = CFArrayGetValueAtIndex(splitter, 0);
			if (isA_CFString(prefixAddrPtr) == NULL) {
				IPConfigLog(LOG_NOTICE, "bad prefix addr type");
				wellFormed = false;
				break;
			}
			prefixAddrStr = (CFStringRef)prefixAddrPtr;
			if (!my_CFStringToIPv6Address(prefixAddrStr, &prefixAddr)) {
				IPConfigLog(LOG_NOTICE, "bad ipv6 address");
				wellFormed = false;
				break;
			}
			prefixLenPtr = CFArrayGetValueAtIndex(splitter, 1);
			if (isA_CFString(prefixLenPtr) == NULL) {
				IPConfigLog(LOG_NOTICE, "bad prefix len type");
				wellFormed = false;
				break;
			}
			prefixLenStr = (CFStringRef)prefixLenPtr;
			if (!my_CFStringToNumber(prefixLenStr, &prefixLenUInt)
			    || prefixLenUInt < 0
			    || prefixLenUInt > 128) {
				IPConfigLog(LOG_NOTICE, "bad prefix len value");
				wellFormed = false;
				break;
			}
			// validated good prefix, adds str 'prefix/prefixlen' to ret array
			copyStr = CFStringCreateWithFormat(NULL, NULL,
							   CFSTR("%@/%@"),
							   prefixAddrStr,
							   prefixLenStr);
			if (copyStr == NULL) {
				wellFormed = false;
				break;
			}
			CFArrayAppendValue(prefixStringsCIDR, copyStr);
			[prefixStrings addObject:(__bridge NSString *)prefixStr];
			my_CFRelease(&splitter);
			my_CFRelease(&copyStr);
		}
	}
	my_CFRelease(&splitter);
	if (!wellFormed) {
		IPConfigLog(LOG_NOTICE, "failed to parse well-formed prefixes");
		prefixStrings = nil;
		my_CFRelease(&prefixStringsCIDR);
		goto done;
	}

#ifndef __TEST_IPH_PVD__
	/*
	 * this checks that each of the prefixes advertised in the RA's
	 * PIOs is included in at least one of the listed prefixes in
	 * the retrieved PvD Additional Info 'prefixes' array
	 */
	// the prefixes array sizes are neglibible; it's ok to check with a double loop
	for (NSString *prefix1 in raPIOPrefixes) {
		NSArray<NSString *> *splitter1 = [prefix1 componentsSeparatedByString:@"::"];
		NSString *prefix1Comparable = [splitter1 firstObject];
		for (NSString *prefix2 in prefixStrings) {
			NSArray<NSString *> *splitter2 = [prefix2 componentsSeparatedByString:@"::"];
			NSString *prefix2Comparable = [splitter2 firstObject];
			// a shorter prefix ffff:: may contain a longer prefix ffff:1::,
			// but a shorter string can't contain a longer one
			// that's why prefix1 and prefix2 are swapped here
			if ([[prefix1Comparable lowercaseString]
			     containsString:[prefix2Comparable lowercaseString]]) {
				IPConfigLog(LOG_DEBUG, "RA PIO prefix '%@' found contained by PvD"
					" Additional Information prefix '%@'",
					prefix1Comparable, prefix2Comparable);
				contained = true;
				break;
			}
		}
		if (!contained) {
			IPConfigLog(LOG_NOTICE, "discrepancy found with prefix '%@', "
				"not contained by any of %@",
				prefix1Comparable, prefixStrings);
			goto done;
		}
		contained = false;
	}
#endif // __TEST_IPH_PVD__

	*prefixes = (CFArrayRef)prefixStringsCIDR;
	IPConfigLog(LOG_DEBUG, "successfully validated prefixes array %@", *prefixes);
	success = true;

done:
	if (*prefixes == NULL) {
		my_CFRelease(&prefixStringsCIDR);
	}
	return success;
}

/*
 * This section validates all NECESSARY keys defined in RFC 8801.
 */
static bool
set_valid_necessary_fields(CFMutableDictionaryRef additionalInfoDict, NSDictionary * parsedJSONDict,
			   NSString * pvdID, NSArray<NSString *> * raPIOPrefixes)
{
	CFStringRef identifier = NULL;
	CFStringRef expires = NULL;
	CFArrayRef prefixes = NULL;
	const NSArray *necessaryFieldsArray = nil;
	bool success = false;

	necessaryFieldsArray = @[@NECESSARY_FIELD_IDENTIFIER, @NECESSARY_FIELD_EXPIRES, @NECESSARY_FIELD_PREFIXES];
	for (NSString *field in necessaryFieldsArray) {
		@autoreleasepool {
			id val = nil;

			/* Fields must be included at the root of the JSON object */
			if ((val = [parsedJSONDict valueForKey:field]) == nil) {
				IPConfigLog(LOG_NOTICE, "PvD Additional Info is missing necessary field '%@'", field);
				goto done;
			}

			if ([field isEqualToString:@NECESSARY_FIELD_IDENTIFIER] && identifier == NULL) {
				if (!copy_valid_identifier(&identifier, field, val, pvdID)) {
					IPConfigLog(LOG_NOTICE, "failed to validate field '%@' : %@", field, val);
					goto done;
				}
			} else if ([field isEqualToString:@NECESSARY_FIELD_EXPIRES] && expires == NULL) {
				if (!copy_valid_expires(&expires, field, val)) {
					IPConfigLog(LOG_NOTICE, "failed to validate field '%@': %@", field, val);
					goto done;
				}
			} else if ([field isEqualToString:@NECESSARY_FIELD_PREFIXES] && prefixes == NULL) {
				if (!copy_valid_prefixes(&prefixes, field, val, raPIOPrefixes)) {
					IPConfigLog(LOG_NOTICE, "failed to validate field '%@' : %@", field, val);
					goto done;
				}
			}
		}
	}
	CFDictionarySetValue(additionalInfoDict, CFSTR(NECESSARY_FIELD_IDENTIFIER), identifier);
	CFDictionarySetValue(additionalInfoDict, CFSTR(NECESSARY_FIELD_EXPIRES), expires);
	CFDictionarySetValue(additionalInfoDict, CFSTR(NECESSARY_FIELD_PREFIXES), prefixes);
	success = (CFDictionaryGetCount(additionalInfoDict) == NECESSARY_FIELDS_COUNT);

done:
	my_CFRelease(&identifier);
	my_CFRelease(&expires);
	my_CFRelease(&prefixes);
	return success;
}

static bool
copy_valid_dns_zones(CFArrayRef * dnsZones, NSString * field, id val)
{
	/* Must be Array<Strings<FQDN>> */
	bool success = false;
	CFMutableArrayRef dnsZonesArray = NULL;
	NSUInteger arrayMaxCount = (([(NSArray *)val count] < COLLECTION_TYPE_FIELD_MAX_COUNT)
				    ? [(NSArray *)val count]
				    : COLLECTION_TYPE_FIELD_MAX_COUNT);

	IPConfigLog(LOG_DEBUG, "validating field '%@' with value '%@' of class '%@'", field, val, [val class]);
	if (![val isKindOfClass:[NSArray class]]) {
		IPConfigLog(LOG_DEBUG, "expected JSON value of Array type");
		goto done;
	}
	dnsZonesArray = CFArrayCreateMutable(NULL, 1, &kCFTypeArrayCallBacks);
	for (NSUInteger i = 0;  i < arrayMaxCount; i++) {
		@autoreleasepool {
			id dnsZoneString = nil;
			CFDataRef isValidFQDN = NULL;
			CFStringRef dnsZone = NULL;

			dnsZoneString = [(NSArray *)val objectAtIndex:i];
			if (![dnsZoneString isKindOfClass:[NSString class]]) {
				IPConfigLog(LOG_NOTICE, "expected String element, got '%@'", [dnsZoneString class]);
				// Throw away whole dnsZones field if any of its elements is malformed.
				goto done;
			}
			if ((isValidFQDN
			     = DNSNameListDataCreateWithString((__bridge CFStringRef)dnsZoneString)) == NULL) {
				IPConfigLog(LOG_DEBUG, "couldn't validate DNS Zone '%@' as an FQDN",
					dnsZoneString);
				goto done;
			}
			my_CFRelease(&isValidFQDN);
			if ((dnsZone
			     = CFStringCreateCopy(NULL, (__bridge CFStringRef)dnsZoneString)) == NULL) {
				goto done;
			}
			CFArrayAppendValue(dnsZonesArray, dnsZone);
			my_CFRelease(&dnsZone);
		}
	}
	*dnsZones = (CFArrayRef)dnsZonesArray;
	IPConfigLog(LOG_DEBUG, "successfully validated DNS Zones array %@", *dnsZones);
	success = true;

done:
	IPConfigLog(LOG_DEBUG, "failed to validate field '%@'", field);
	if (dnsZonesArray == NULL) {
		my_CFRelease(&dnsZonesArray);
	}
	return success;
}

static bool
copy_valid_no_internet(CFBooleanRef * noInternet, NSString * field, id val)
{
	/* Must be Boolean */
	bool success = false;

	IPConfigLog(LOG_DEBUG, "validating field '%@' with value '%@' of class '%@'", field, val, [val class]);
	if (![val isKindOfClass:[NSString class]]) {
		IPConfigLog(LOG_NOTICE, "expected String element, got '%@'", [val class]);
		goto done;
	}
	if ([val isEqualToString:@"true"]) {
		*noInternet = kCFBooleanTrue;
	} else if ([val isEqualToString:@"false"]) {
		*noInternet = kCFBooleanFalse;
	} else {
		goto done;
	}
	success = true;

done:
	return success;
}

/*
 * This adds key-vals that are defined in RFC 8801 but may or may not
 * be included in a valid PvD JSON object (i.e. OPTIONAL).
 * PvD additional info retrieval is considered a success even if
 * any of these OPTIONAL fields are missing from the JSON object.
 */
static bool
set_valid_optional_fields(CFMutableDictionaryRef additionalInfoDict, NSDictionary * parsedJSONDict)
{
	CFArrayRef dnsZones = NULL;
	CFBooleanRef noInternet = NULL;
	const NSArray *optionalFieldsArray = nil;

	optionalFieldsArray = @[@OPTIONAL_FIELD_DNS_ZONES, @OPTIONAL_FIELD_NO_INTERNET];
	for (NSString *field in optionalFieldsArray) {
		@autoreleasepool {
			id val = nil;

			if ((val = [parsedJSONDict valueForKey:field]) == nil) {
				/* checks whether ANY optional fields are included */
				continue;
			}
			if ([field isEqualToString:@OPTIONAL_FIELD_DNS_ZONES] && dnsZones == NULL) {
				if (!copy_valid_dns_zones(&dnsZones, field, val)) {
					IPConfigLog(LOG_NOTICE, "failed to validate field '%@' : %@", field, val);
					continue;
				}
			} else if ([field isEqualToString:@OPTIONAL_FIELD_NO_INTERNET] && noInternet == NULL) {
				if (!copy_valid_no_internet(&noInternet, field, val)) {
					IPConfigLog(LOG_NOTICE, "failed to validate field '%@' : %@", field, val);
					continue;
				}
			}
		}
	}
	if (dnsZones != NULL) {
		CFDictionarySetValue(additionalInfoDict, CFSTR(OPTIONAL_FIELD_DNS_ZONES), dnsZones);
		my_CFRelease(&dnsZones);
	}
	if (noInternet != NULL) {
		CFDictionarySetValue(additionalInfoDict, CFSTR(OPTIONAL_FIELD_NO_INTERNET), noInternet);
		my_CFRelease(&noInternet);
	}
	return true;
}

static bool
copy_valid_proxies(CFArrayRef * proxies, NSString * field, id val)
{
	/* Expects Array<String> or Array<Dictionary<String,<T>>> */
	bool success = false;
	CFMutableArrayRef mutableProxies = NULL;
	NSUInteger arrayMaxCount = (([(NSArray *)val count] < COLLECTION_TYPE_FIELD_MAX_COUNT)
				    ? [(NSArray *)val count]
				    : COLLECTION_TYPE_FIELD_MAX_COUNT);

	IPConfigLog(LOG_DEBUG, "validating field '%@' with value '%@' of class '%@'", field, val, [val class]);
	if (![val isKindOfClass:[NSArray class]]) {
		IPConfigLog(LOG_NOTICE, "expected JSON value of Array type");
		goto done;
	}
	mutableProxies = CFArrayCreateMutable(NULL, 0, &kCFTypeArrayCallBacks);
	if (mutableProxies == NULL) {
		goto done;
	}
	for (NSUInteger i = 0;  i < arrayMaxCount; i++) {
		@autoreleasepool {
			id proxy = nil;

			proxy = [(NSArray *)val objectAtIndex:i];
			if ([proxy isKindOfClass:[NSString class]]) {
				CFArrayAppendValue(mutableProxies, (__bridge CFStringRef)proxy);
			} else if ([proxy isKindOfClass:[NSDictionary class]]) {
				CFMutableDictionaryRef proxySubdict = NULL;

				proxySubdict = CFDictionaryCreateMutable(NULL, 0,
									 &kCFTypeDictionaryKeyCallBacks,
									 &kCFTypeDictionaryValueCallBacks);
				if (proxySubdict == NULL) {
					goto done;
				}
				for (NSString *subdictKey in proxy) {
					NSObject *subdictVal = [(NSDictionary *)proxy objectForKey:subdictKey];

					CFDictionaryAddValue(proxySubdict, (__bridge CFStringRef)subdictKey,
							     (__bridge CFTypeRef)subdictVal);
				}
				CFArrayAppendValue(mutableProxies, (CFDictionaryRef)proxySubdict);
				my_CFRelease(&proxySubdict);
			} else {
				IPConfigLog(LOG_NOTICE, "expected String or Dictionary element, got '%@'", [proxy class]);
				goto done;
			}
		}
	}
	*proxies = (CFArrayRef)mutableProxies;
	IPConfigLog(LOG_DEBUG, "got proxies array:\n%@", *proxies);
	success = true;

done:
	if (*proxies == NULL) {
		my_CFRelease(&mutableProxies);
	}
	return success;
}

static bool
element_is_cfprimitive(id val)
{
	CFTypeRef valcf = (__bridge CFTypeRef)val;
	bool res = false;

	if (valcf == NULL) {
		goto done;
	}
	if (isA_CFBoolean(valcf)
	    || isA_CFNumber(valcf)
	    || isA_CFString(valcf)
	    || isA_CFDate(valcf)
	    || isA_CFData(valcf)) {
		res = true;
	}

done:
	if (!res) {
		IPConfigLog(LOG_DEBUG, "element is not a cfprimitive: %@", val);
	}
	return res;
}

static bool
element_is_cfcollection(id val)
{
	CFTypeRef valcf = (__bridge CFTypeRef)val;
	bool res = false;

	if (valcf == NULL) {
		goto done;
	}
	if (isA_CFArray(valcf)
	    || isA_CFDictionary(valcf)) {
		res = true;
	}

done:
	if (!res) {
		IPConfigLog(LOG_DEBUG, "element is not a cfcollection: %@", val);
	}
	return res;
}

static bool
enforce_proper_boundaries(NSArray * elements, int level)
{
	bool success = false;

	if (elements == nil) {
		goto done;
	}
	if ([elements count] > ELEMENTS_PER_LEVEL_MAX) {
		goto done;
	}
	if (level == NESTING_LEVEL_MAX) {
		/* no more nested structures allowed */
		for (id e in elements) {
			if (!element_is_cfprimitive(e)) {
				goto done;
			}
		}
	} else {
		for (id e in elements) {
			CFTypeRef cfelement = NULL;
			NSArray * nested_elements = nil;

			if (element_is_cfprimitive(e)) {
				continue;
			}
			if (!element_is_cfcollection(e)) {
				goto done;
			}
			cfelement = (__bridge CFTypeRef)e;
			if (isA_CFArray(cfelement)) {
				nested_elements = e;
			} else if (isA_CFDictionary(cfelement)) {
				nested_elements = [e allValues];
			}
			if (!enforce_proper_boundaries(nested_elements, level+1)) {
				goto done;
			}
		}

	}
	success = true;

done:
	if (!success) {
		IPConfigLog(LOG_DEBUG, "%s: failed boundary checks at nesting level %d for elements array %@",
			    __func__, level, elements);
	}
	return success;
}

/*
 * This adds key-vals that aren't defined in RFC 8801 (i.e. EXTRA).
 * PvD additional info retrieval is considered a success even if
 * any of these EXTRA fields are missing from the JSON object.
 */
static bool
set_valid_extra_fields(CFMutableDictionaryRef additionalInfoDict, NSDictionary * parsedJSONDict)
{
	CFArrayRef proxies = NULL;
	const NSArray *extraFieldsArray = nil;
	NSArray * knownKeys = nil;
	NSUInteger allKeysCount = 0;

	extraFieldsArray = @[@EXTRA_FIELD_PROXIES];
	for (NSString *field in extraFieldsArray) {
		@autoreleasepool {
			id val = nil;

			if ((val = [parsedJSONDict valueForKey:field]) == nil) {
				continue;
			}
			if ([field isEqualToString:@EXTRA_FIELD_PROXIES] && proxies == NULL) {
				if (!copy_valid_proxies(&proxies, field, val)) {
					IPConfigLog(LOG_NOTICE, "failed to validate field '%@' : %@", field, val);
					continue;
				}
			}
		}
	}
	if (proxies != NULL) {
		CFDictionarySetValue(additionalInfoDict, CFSTR(EXTRA_FIELD_PROXIES), proxies);
		my_CFRelease(&proxies);
	}
	/* this adds any other extra fields that we don't recognize (may be experimental or privately-deployed) */
	knownKeys = [(__bridge NSDictionary *)additionalInfoDict allKeys];
	allKeysCount = [knownKeys count];
	for (NSString * field in [parsedJSONDict allKeys]) {
		id val = NULL;

		if (allKeysCount >= ELEMENTS_PER_LEVEL_MAX) {
			break;
		}
		if ([knownKeys containsObject:field]) {
			continue;
		}
		val = [parsedJSONDict valueForKey:field];
		if (!enforce_proper_boundaries(@[val], 0)) {
			IPConfigLog(LOG_NOTICE, "failed to validate field '%@' : %@", field, val);
			continue;
		}
		CFDictionarySetValue(additionalInfoDict, (__bridge CFStringRef)field, (__bridge CFTypeRef)val);
		allKeysCount++;
	}

	return true;
}

- (CFDictionaryRef)createValidPvDAdditionalInfoDict:(NSDictionary *)parsedJSONDict
					     withID:(NSString *)pvdID
					andPrefixes:(NSArray<NSString *> *)raPIOPrefixes
CF_RETURNS_RETAINED
{
	CFMutableDictionaryRef additionalInfoDict = NULL;
	NSArray * additionalInfoDictValues = nil;
	bool success = false;

	if (parsedJSONDict == nil) {
		IPConfigLog(LOG_NOTICE, "can't create valid info dict from empty JSON");
		goto done;
	}

	/* Initializes and makes CFDictionary to be sent in the XPC reply */
	additionalInfoDict = CFDictionaryCreateMutable(NULL, NECESSARY_FIELDS_COUNT,
						       &kCFTypeDictionaryKeyCallBacks,
						       &kCFTypeDictionaryValueCallBacks);
	if (additionalInfoDict == NULL) {
		IPConfigLog(LOG_NOTICE, "failed to create additional info CFDictionary");
		goto done;
	}
	if (!set_valid_necessary_fields(additionalInfoDict, parsedJSONDict, pvdID, raPIOPrefixes)
	    || !set_valid_optional_fields(additionalInfoDict, parsedJSONDict)
	    || !set_valid_extra_fields(additionalInfoDict, parsedJSONDict)) {
		goto done;
	}
	if (additionalInfoDict == NULL || CFDictionaryGetCount(additionalInfoDict) < NECESSARY_FIELDS_COUNT) {
		goto done;
	}
	/*
	 * In the case of there being unknown keys in the retrieved JSON object,
	 * the retrieval is still considered successful. This allows up to 10 total keys,
	 * accounting for the recognized ones first. Each key can either be a primitive-like
	 * CFType, i.e. String or Number, or a collection of CFTypes. If the value is a collection,
	 * it can only contain elements that are primitive-like, or flat collections,
	 * i.e. Array<V> or Dictionary<K:V>, where V is any of { Boolean, Number, String, Data, Date }.
	 * Another way to put this: top-level values are allowed at maximum 2 levels of nesting.
	 */
	additionalInfoDictValues = [(__bridge NSDictionary *)additionalInfoDict allValues];
	if (!enforce_proper_boundaries(additionalInfoDictValues, 0)) {
		goto done;
	}
	success = true;

done:
	if (!success) {
		IPConfigLog(LOG_NOTICE, "validation failed, couldn't create PvD Additional Info dictionary");
		self.validFetch = NO;
	} else {
		/* SUCCESS */
		IPConfigLog(LOG_DEBUG, "validation succeeded, got PvD Additional Info dict:\n%@", additionalInfoDict);
	}
	return (CFDictionaryRef)additionalInfoDict;
}

@end

#ifdef TESTHARNESS_IPHPVD

int
main(int argc, char * argv[])
{
	bool res = true;
	NSDictionary * test_dict_good = @{
		@NECESSARY_FIELD_IDENTIFIER : @"test-domain.local",
		@EXTRA_FIELD_PROXIES : @[
			@{
				@"test_key" : @"test_val"
			},
			@{
				@"test_key2" : @"test_val2"
			}
		]
	};
	NSDictionary * test_dict_bad1 = @{
		@NECESSARY_FIELD_IDENTIFIER : @"test-domain.local",
		@EXTRA_FIELD_PROXIES : @[
			@{ @"test_key" : @"test_val" },
			@{ @"test_key" : @"test_val" },
			@{ @"test_key" : @"test_val" },
			@{ @"test_key" : @"test_val" },
			@{ @"test_key" : @"test_val" },
			@{ @"test_key" : @"test_val" },
			@{ @"test_key" : @"test_val" },
			@{ @"test_key" : @"test_val" },
			@{ @"test_key" : @"test_val" },
			@{ @"test_key" : @"test_val" },
			@{ @"test_key" : @"test_val" }, // 11 count
		]
	};
	NSDictionary * test_dict_bad2 = @{
		@"test_field" : @[ @[ @[ @"test_nest" ] ] ] // 3-deep nesting
	};

	res = enforce_proper_boundaries([test_dict_good allValues], 0);
	if (!res) {
		printf("failed test 1");
		goto done;
	}
	res = enforce_proper_boundaries([test_dict_bad1 allValues], 0);
	if (res) {
		printf("failed test 2");
		goto done;
	}
	res = enforce_proper_boundaries([test_dict_bad2 allValues], 0);
	if (res) {
		printf("failed test 3");
		goto done;
	}

done:
	return res;
}

#endif // TESTHARNESS_IPHPVD
