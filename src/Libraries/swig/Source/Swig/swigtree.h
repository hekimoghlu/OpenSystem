/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 26, 2022.
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
/* $Id: swig.h 9622 2006-12-19 03:49:17Z beazley $ */

/* Macros to traverse the DOM tree */

#define  nodeType(x)               Getattr(x,"nodeType")
#define  parentNode(x)             Getattr(x,"parentNode")
#define  previousSibling(x)        Getattr(x,"previousSibling")
#define  nextSibling(x)            Getattr(x,"nextSibling")
#define  firstChild(x)             Getattr(x,"firstChild")
#define  lastChild(x)              Getattr(x,"lastChild")

/* Macros to set up the DOM tree (mostly used by the parser) */

#define  set_nodeType(x,v)         Setattr(x,"nodeType",v)
#define  set_parentNode(x,v)       Setattr(x,"parentNode",v)
#define  set_previousSibling(x,v)  Setattr(x,"previousSibling",v)
#define  set_nextSibling(x,v)      Setattr(x,"nextSibling",v)
#define  set_firstChild(x,v)       Setattr(x,"firstChild",v)
#define  set_lastChild(x,v)        Setattr(x,"lastChild",v)

/* Utility functions */

extern int    checkAttribute(Node *obj, const_String_or_char_ptr name, const_String_or_char_ptr value);
extern void   appendChild(Node *node, Node *child);
extern void   prependChild(Node *node, Node *child);
extern void   removeNode(Node *node);
extern Node  *copyNode(Node *node);

/* Node restoration/restore functions */

extern void  Swig_require(const char *ns, Node *node, ...);
extern void  Swig_save(const char *ns, Node *node, ...);
extern void  Swig_restore(Node *node);

/* Debugging of parse trees */

extern void Swig_print_tags(File *obj, Node *root);
extern void Swig_print_tree(Node *obj);
extern void Swig_print_node(Node *obj);
