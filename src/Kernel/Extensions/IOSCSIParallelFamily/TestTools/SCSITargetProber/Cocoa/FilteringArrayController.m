/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 19, 2024.
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
//ÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑ
//	Imports
//ÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑ

#import "FilteringArrayController.h"
#import "SCSITargetProberKeys.h"
#import <Foundation/NSKeyValueObserving.h>


//ÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑ
//	Implementation
//ÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑÑ

@implementation FilteringArrayController


// Called when the SCSITargetProberDocument nib loads.

- ( void ) awakeFromNib
{
	
	// Initialize the search category to be by title (Description).
	searchCategory = kSearchCategoryTitle;
	
	// Set the placeholder string for the search category.
	[ [ searchField cell ] setPlaceholderString: [ self placeholderForTag: searchCategory ] ];
	
	// Setup the search menu.
	[ self setupSearchMenu ];
	
	// Observe any value changes to NSUserDefaultsController.
	[ [ NSUserDefaultsController sharedUserDefaultsController ] addObserver: self
		forKeyPath: kShowTargetIDKeyPath options: NSKeyValueObservingOptionNew context: nil ];

	[ [ NSUserDefaultsController sharedUserDefaultsController ] addObserver: self
		forKeyPath: kShowDescriptionKeyPath options: NSKeyValueObservingOptionNew context: nil ];

	[ [ NSUserDefaultsController sharedUserDefaultsController ] addObserver: self
		forKeyPath: kShowRevisionKeyPath options: NSKeyValueObservingOptionNew context: nil ];

	[ [ NSUserDefaultsController sharedUserDefaultsController ] addObserver: self
		forKeyPath: kShowFeaturesKeyPath options: NSKeyValueObservingOptionNew context: nil ];

	[ [ NSUserDefaultsController sharedUserDefaultsController ] addObserver: self
		forKeyPath: kShowPDTKeyPath options: NSKeyValueObservingOptionNew context: nil ];
	
}


// Called when values change for NSUserDefaultsController.
// See NSKeyValueObserving.h for more information.

- ( void ) observeValueForKeyPath: ( NSString * ) keyPath
						 ofObject: ( id ) object
						   change: ( NSDictionary * ) change
						  context: ( void * ) context
{

#pragma unused ( keyPath )
#pragma unused ( object )
#pragma unused ( change )
#pragma unused ( context )
	
	// Set up the search menu based on any changes observed.
	[ self setupSearchMenu ];
	
}


// Get a localized string to use as a placeholder in the NSSearchFieldCell
// based on the passed in tag (which corresponds to the selected menu item
// in the NSSearchFieldCell's menu).

- ( NSString * ) placeholderForTag: ( int ) tag
{
	
	NSString *	string = nil;
	
	switch ( tag )
	{
		
		case kSearchCategoryID:
			string = kIDString;
			break;
		
		case kSearchCategoryTitle:
			string = kDescriptionString;
			break;
		
		case kSearchCategoryRevision:
			string = kRevisionString;
			break;
		
		case kSearchCategoryFeatures:
			string = kFeaturesString;
			break;
		
		case kSearchCategoryPDT:
			string = kPDTString;
			break;
		
	}
	
	// Get the localized string based on the string we have.
	string = [ [ NSBundle mainBundle ] localizedStringForKey: string
													   value: string
													   table: nil ];
	
	return string;
	
}


// Called to set up the search menu (removes items not
// checked in prefs box and sets checkmark on selected category).

- ( void ) setupSearchMenu
{
	
	NSMenu *	menu	= nil;
	id			values	= nil;
	id			item 	= nil;
	int			index	= 0;
	int			count	= 0;
	BOOL		itemSet	= NO;
	
	// Copy the menu in the nib.
	menu = [ sortMenu copyWithZone: nil ];
	
	values = [ [ NSUserDefaultsController sharedUserDefaultsController ] values ];
	
	// Should we remove the "ID" category from the search menu?
	if ( [ [ values valueForKey: kShowTargetIDString ] boolValue ] == NO )
	{
		
		// Yes, remove it.
		item = [ menu itemWithTag: kSearchCategoryID ];
		[ menu removeItem: item ];
		
	}
	
	// Should we remove the "Description" category from the search menu?
	if ( [ [ values valueForKey: kShowDescriptionString ] boolValue ] == NO )
	{
		
		// Yes, remove it.
		item = [ menu itemWithTag: kSearchCategoryTitle ];
		[ menu removeItem: item ];
		
	}
	
	// Should we remove the "Revision" category from the search menu?
	if ( [ [ values valueForKey: kShowRevisionString ] boolValue ] == NO )
	{
		
		// Yes, remove it.
		item = [ menu itemWithTag: kSearchCategoryRevision ];
		[ menu removeItem: item ];
		
	}
	
	// Should we remove the "Features" category from the search menu?
	if ( [ [ values valueForKey: kShowFeaturesString ] boolValue ] == NO )
	{
		
		// Yes, remove it.
		item = [ menu itemWithTag: kSearchCategoryFeatures ];
		[ menu removeItem: item ];
		
	}
	
	// Should we remove the "PDT" category from the search menu?
	if ( [ [ values valueForKey: kShowPDTString ] boolValue ] == NO )
	{
		
		// Yes, remove it.
		item = [ menu itemWithTag: kSearchCategoryPDT ];
		[ menu removeItem: item ];
		
	}
	
	count = [ menu numberOfItems ];
	for ( index = 0; index < count; index++ )
	{
		
		item = [ menu itemAtIndex: index ];
		
		// Is the item with this tag the one we're currently using as
		// our search category?
		if ( [ item tag ] != searchCategory )
		{
			
			// No, make sure checkmark is off.
			[ item setState: NSOffState ];
			
		}
		
		else
		{
			
			// Yes, make sure checkmark is on.
			[ item setState: NSOnState ];
			
			// Make sure we mark that an item was set.
			itemSet = YES;
			
		}
		
	}
	
	// If there wasn't an item picked (because the preferences removed it), but there are
	// still elements in the list, change the search category to the first available one.
	if ( ( itemSet == NO ) && ( count > 0 ) )
	{
		[ [ searchField cell ] setStringValue: @"" ];
		[ self changeSearchCategory: [ menu itemAtIndex: 0 ] ];
	}
	
	else
	{
		
		// Set the search menu template. If we didn't get into this else, then
		// the search category will get changed and we will come back and hit
		// this case for sure.
		[ [ searchField cell ] setSearchMenuTemplate: menu ];
		
	}
	
}


// Action called by NSSearchField when something has been typed in it.

- ( IBAction ) search: ( id ) sender
{
	
	// Set the search string so we can use it in arrangeObjects:
	searchString = [ sender stringValue ];
	
	// Call rearrange objects. This decides to call arrangeObjects or not.
	[ self rearrangeObjects ];
	
}


// Called by all menu items in the NSMenu attached to the NSSearchFieldCell
// to change the search category.

- ( IBAction ) changeSearchCategory: ( id ) sender
{
	
	// Get the new category by the tag from the menu item which sent the
	// action.
	searchCategory = [ sender tag ];
	
	// Make sure the UI updates the visible items based on the search
	// category change.
	[ self search: searchField ];
	
	// Change the placeholder string so that when the NSSearchField is cleared
	// or inactive, a placeholder exists.
	[ [ searchField cell ] setPlaceholderString: [ self placeholderForTag: searchCategory ] ];
	
	// Change the search menu (changes the checkmarks).
	[ self setupSearchMenu ];
		
}


// Called to arrange objects.

- ( NSArray * ) arrangeObjects: ( NSArray * ) objects
{
	
	// Is there a search string?
	if ( ( searchString == nil ) || ( [ searchString isEqualToString: @"" ] ) )
	{
		// No, short circuit out...
		return [ super arrangeObjects: objects ];
	}
	
	NSMutableArray *	filteredObjects		= nil;
	NSEnumerator *		objectsEnumerator	= nil;
	NSString *			keyPath				= nil;
	id 					item				= nil;
	
	// Create an array of the same size as the original (so it's big
	// enough to hold all the potential filtered objects)
	filteredObjects = [ NSMutableArray arrayWithCapacity: [ objects count ] ];
	
	// Figure out which key path to use based on the search category.
	switch ( searchCategory )
	{
		
		case kSearchCategoryID:
			keyPath = kDeviceIdentifierKeyPath;
			break;
		
		case kSearchCategoryTitle:
			keyPath = kDeviceTitleKeyPath;
			break;
		
		case kSearchCategoryRevision:
			keyPath = kDeviceRevisionKeyPath;
			break;
		
		case kSearchCategoryFeatures:
			keyPath = kDeviceFeaturesKeyPath;
			break;
		
		case kSearchCategoryPDT:
			keyPath = kDevicePDTKeyPath;
			break;
		
	}
	
	// Get an object enumerator
	objectsEnumerator = [ objects objectEnumerator ];
	
	// Loop over the items and based on the search category, figure out if the
	// item should remain visible in the table view or not.
	item = [ objectsEnumerator nextObject ];
	while ( item != nil )
	{
		
		NSString * 	lowerCaseSearchString 	= nil;
		id			value					= nil;
		
		// Change to lower case string for searching.
		lowerCaseSearchString = [ searchString lowercaseString ];
		
		// Get the value at the key path.
		value = [ item valueForKeyPath: keyPath ];
		
		// Add the object if it matches.
		[ self addObject: item toArray: filteredObjects ifValue: value matchesString: lowerCaseSearchString ];
		
		// Get the next object.
		item = [ objectsEnumerator nextObject ];
		
	}
	
	return [ super arrangeObjects: filteredObjects ];
	
}


// Adds the object to the array if the matching string matches.

- ( void ) addObject: ( id ) object
			 toArray: ( NSMutableArray * ) array
			 ifValue: ( id ) inValue
	   matchesString: ( NSString * ) matchingString
{
	
	// Is the value an NSArray of values?
	if ( [ inValue isKindOfClass: [ NSArray class ] ] )
	{			
		
		NSArray *	values 		= nil;
		int			count		= 0;
		int			index		= 0;
		
		// Yes, we have an NSArray of values. We have to loop over the whole array to
		// determine if the item should be visible or not.
		
		values = ( NSArray * ) inValue;
		count = [ values count ];
		
		for ( index = 0; index < count; index++ )
		{
			
			id	value = nil;
			
			// Get the object at index.
			value = [ values objectAtIndex: index ];
			
			// Call this method recursively (the object at that index could be an NSNumber,
			// NSString, NSArray, etc.)
			[ self addObject: object toArray: array ifValue: value matchesString: matchingString ];
			
		}
		
	}
	
	// Is this value an NSNumber?
	else if ( [ inValue isKindOfClass: [ NSNumber class ] ] )
	{
		
		NSString *  string	= nil;
		
		string = [ [ inValue stringValue ] lowercaseString ];
		
		// Is the search string anywhere to be found?
		if ( [ string rangeOfString: matchingString ].location != NSNotFound )
		{
			
			// Yes, add the item to the filtered objects list.
			[ array addObject: object ];
				
		}
	
	}
	
	// Is this value an NSString?
	else if ( [ inValue isKindOfClass: [ NSString class ] ] )
	{	
		
		NSString *  string	= nil;
		
		string = [ inValue lowercaseString ];
		
		// Is the search string anywhere to be found?
		if ( [ string rangeOfString: matchingString ].location != NSNotFound )
		{
			
			// Yes, add the item to the filtered objects list.
			[ array addObject: object ];
				
		}
		
	}
	
	// Add more class types as necessary...
	
}


@end