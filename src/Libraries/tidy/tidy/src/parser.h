/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 16, 2022.
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

#ifndef __PARSER_H__
#define __PARSER_H__

/* parser.h -- HTML Parser

  (c) 1998-2006 (W3C) MIT, ERCIM, Keio University
  See tidy.h for the copyright notice.
  
  CVS Info :

    $Author$ 
    $Date$ 
    $Revision$ 

*/

#include "forward.h"

Bool TY_(CheckNodeIntegrity)(Node *node);

/*
 used to determine how attributes
 without values should be printed
 this was introduced to deal with
 user defined tags e.g. Cold Fusion
*/
Bool TY_(IsNewNode)(Node *node);

void TY_(CoerceNode)(TidyDocImpl* doc, Node *node, TidyTagId tid, Bool obsolete, Bool expected);

/* extract a node and its children from a markup tree */
Node *TY_(RemoveNode)(Node *node);

/* remove node from markup tree and discard it */
Node *TY_(DiscardElement)( TidyDocImpl* doc, Node *element);

/* insert node into markup tree as the firt element
 of content of element */
void TY_(InsertNodeAtStart)(Node *element, Node *node);

/* insert node into markup tree as the last element
 of content of "element" */
void TY_(InsertNodeAtEnd)(Node *element, Node *node);

/* insert node into markup tree before element */
void TY_(InsertNodeBeforeElement)(Node *element, Node *node);

/* insert node into markup tree after element */
void TY_(InsertNodeAfterElement)(Node *element, Node *node);

Node *TY_(TrimEmptyElement)( TidyDocImpl* doc, Node *element );
Node* TY_(DropEmptyElements)(TidyDocImpl* doc, Node* node);


/* assumes node is a text node */
Bool TY_(IsBlank)(Lexer *lexer, Node *node);

Bool TY_(IsJavaScript)(Node *node);

/*
  HTML is the top level element
*/
void TY_(ParseDocument)( TidyDocImpl* doc );



/*
  XML documents
*/
Bool TY_(XMLPreserveWhiteSpace)( TidyDocImpl* doc, Node *element );

void TY_(ParseXMLDocument)( TidyDocImpl* doc );

#endif /* __PARSER_H__ */
