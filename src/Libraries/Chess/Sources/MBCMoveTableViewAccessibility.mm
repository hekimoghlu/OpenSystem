/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 13, 2024.
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
#import "MBCMoveTableViewAccessibility.h"
#import "MBCMoveTableView.h"

@implementation MBCMoveAccessibilityProxy

+ (id) proxyWithInfo:(MBCGameInfo *)info move:(int)move
{
	return [[[MBCMoveAccessibilityProxy alloc] 
				initWithInfo:info move:move]
			   autorelease];
}

- (id) initWithInfo:(MBCGameInfo *)info move:(int)move
{
	fInfo	= info;
	fMove   = move;

	return self;
}

- (BOOL) isEqual:(MBCMoveAccessibilityProxy *)other
{
	return [other isKindOfClass:[MBCMoveAccessibilityProxy class]]
		&& fInfo == other->fInfo && fMove == other->fMove;
}

- (NSUInteger)hash {
    // Equal objects must hash the same.
    return [fInfo hash] + fMove;
}

- (NSString *) description
{
	return [NSString stringWithFormat:@"Move %d", fMove];
}

- (NSArray *)accessibilityAttributeNames 
{
	return [NSArray arrayWithObjects:
					NSAccessibilityRoleAttribute,
					NSAccessibilityRoleDescriptionAttribute,
					NSAccessibilityParentAttribute,
					NSAccessibilityWindowAttribute,
					NSAccessibilityPositionAttribute,
					NSAccessibilitySizeAttribute,
					NSAccessibilityTitleAttribute,
                   NSAccessibilityDescriptionAttribute,
					NSAccessibilityFocusedAttribute,
					NSAccessibilityEnabledAttribute,
					NSAccessibilityTopLevelUIElementAttribute,
					nil];
}

- (NSArray *)accessibilityActionNames
{
    return [NSArray array];
}

- (id)accessibilityFocusedUIElement
{
	return self;
}

- (BOOL)accessibilityIsIgnored
{
	return NO;
}

- (NSRect)accessibilityFocusRingBounds
{
    NSRect  rect = [[fInfo moveList] rectOfRow:fMove-1];
    rect         = [[fInfo moveList] convertRect:rect toView:nil];
    
    return [[[fInfo moveList] window] convertRectToScreen:rect];
}

- (id)accessibilityAttributeValue:(NSString *)attribute 
{
 	if ([attribute isEqual:NSAccessibilityParentAttribute])
		return [fInfo moveList];
	else if ([attribute isEqual:NSAccessibilityChildrenAttribute])
		return [NSArray array];
	else if ([attribute isEqual:NSAccessibilityWindowAttribute])
		return [[fInfo moveList] window];
	else if ([attribute isEqual:NSAccessibilityRoleAttribute])
		return NSAccessibilityStaticTextRole;
	else if ([attribute isEqual:NSAccessibilityRoleDescriptionAttribute])
		return NSAccessibilityRoleDescription(NSAccessibilityStaticTextRole, nil);
	else if ([attribute isEqual:NSAccessibilityPositionAttribute])
		return [NSValue valueWithPoint:
							[self accessibilityFocusRingBounds].origin];
	else if ([attribute isEqual:NSAccessibilitySizeAttribute])
		return [NSValue valueWithSize:
							[self accessibilityFocusRingBounds].size];
	else if ([attribute isEqual:NSAccessibilityTitleAttribute])
		return [fInfo describeMove:fMove];
	else if ([attribute isEqual:NSAccessibilityValueAttribute])
		return nil;
	else if ([attribute isEqual:NSAccessibilityDescriptionAttribute])
		return @"";
	else if ([attribute isEqual:NSAccessibilityFocusedAttribute])
		return [NSNumber numberWithBool:
							 [[NSApp accessibilityFocusedUIElement] 
								 isEqual:self]];
	else if ([attribute isEqual:NSAccessibilityEnabledAttribute])
		return [NSNumber numberWithBool:YES];
	else if ([attribute isEqual:NSAccessibilityTopLevelUIElementAttribute])
		return [[fInfo moveList] window];
#if 0
	else
		NSLog(@"unknown attr: %@\n", attribute);
#endif

	return nil;
}

- (BOOL)accessibilityIsAttributeSettable:(NSString *)attribute 
{
	if ([attribute isEqual:NSAccessibilityFocusedAttribute])
		return YES;

	return NO;
}

- (void)accessibilitySetValue:(id)value forAttribute:(NSString *)attribute 
{
}

@end

@implementation MBCMoveTableView ( Accessibility )


- (NSArray *)accessibilityAttributeNames 
{
	return [NSArray arrayWithObjects:
            NSAccessibilityRoleAttribute,
            NSAccessibilityRoleDescriptionAttribute,
            NSAccessibilityParentAttribute,
            NSAccessibilityChildrenAttribute,
            NSAccessibilityContentsAttribute,
            NSAccessibilityWindowAttribute,
            NSAccessibilityPositionAttribute,
            NSAccessibilitySizeAttribute,
            NSAccessibilityTopLevelUIElementAttribute,
            NSAccessibilitySelectedChildrenAttribute,
            NSAccessibilityDescriptionAttribute,
            nil];
}

- (NSArray *)accessibilityActionNames
{
    return [NSArray array];
}

- (NSString *)accessibilityRoleAttribute 
{
    return NSAccessibilityGroupRole;
}

- (NSArray *)accessibilityChildrenAttribute 
{
	NSInteger           numMoves    = [self numberOfRows];
    NSMutableArray *    kids        = [NSMutableArray arrayWithCapacity:numMoves];
	for (NSInteger move = 0; move++ < numMoves; )
		[kids addObject: [MBCMoveAccessibilityProxy proxyWithInfo:[self dataSource]
													 move:move]];
    
	return kids;
}

- (NSArray *)accessibilitySelectedChildrenAttribute
{
    return [NSArray arrayWithObject:
            [MBCMoveAccessibilityProxy
             proxyWithInfo:[self dataSource] move:[self selectedRow]+1]];
}

- (id)accessibilityAttributeValue:(NSString *)attribute 
{
	if ([attribute isEqual:NSAccessibilityChildrenAttribute] || [attribute isEqual:NSAccessibilityContentsAttribute]) {
		return [self accessibilityChildrenAttribute];
    } else if ([attribute isEqual:NSAccessibilitySelectedChildrenAttribute]) {
        return [self accessibilitySelectedChildrenAttribute];
    } else if ([attribute isEqual:NSAccessibilityDescriptionAttribute]) {
        return NSLocalizedStringFromTable(@"move_table_desc", @"Spoken", @"Moves");
    } else {
        return [super accessibilityAttributeValue:attribute];
    }
}

- (NSUInteger)accessibilityIndexOfChild:(id)child
{
    if ([child isKindOfClass:[MBCMoveAccessibilityProxy class]]) {
        MBCMoveAccessibilityProxy * moveProxy = (MBCMoveAccessibilityProxy *)child;
        
        if (moveProxy->fInfo == [self dataSource])
            return moveProxy->fMove-1;
        else 
            return NSNotFound;
    }
    return [super accessibilityIndexOfChild:child];
}

- (NSUInteger)accessibilityArrayAttributeCount:(NSString *)attribute
{
    if ([attribute isEqual:NSAccessibilityChildrenAttribute])
        return [self numberOfRows];
    else 
        return [super accessibilityArrayAttributeCount:attribute];
}

- (NSArray *)accessibilityArrayAttributeValues:(NSString *)attribute index:(NSUInteger)index maxCount:(NSUInteger)maxCount
{
    if ([attribute isEqual:NSAccessibilityChildrenAttribute]) {
        NSUInteger numKids = [self numberOfRows];
        NSMutableArray *    kids        = [NSMutableArray arrayWithCapacity:numKids];
        while (index++ < numKids && maxCount--)
            [kids addObject: [MBCMoveAccessibilityProxy proxyWithInfo:[self dataSource]
                                                                 move:index]];
        return kids;
    } else {
        return [super accessibilityArrayAttributeValues:attribute index:index maxCount:maxCount];
    }
}
            
- (BOOL)accessibilityIsIgnored
{
	return NO;
}

- (id)accessibilityHitTest:(NSPoint)point
{
    NSInteger   move = [self rowAtPoint:point];
    id          hit;
	if (move < 0) 
		hit = self;
	else 
		hit = [MBCMoveAccessibilityProxy proxyWithInfo:[self dataSource] move:move+1];

	return hit;
}

- (void)accessibilityPostNotification:(NSString *)notification 
{
    //
    // We are a group, groups have children, not rows
    //
    if ([notification isEqual:NSAccessibilitySelectedRowsChangedNotification])
        notification = NSAccessibilitySelectedChildrenChangedNotification;
    [super accessibilityPostNotification:notification];
}

@end

// Local Variables:
// mode:ObjC
// End:
