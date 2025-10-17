/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 9, 2024.
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
**
**  NAME:
**
**      NAMDUMP.C
**
**  FACILITY:
**
**      Interface Definition Language (IDL) Compiler
**
**  ABSTRACT:
**
**  This is the dumper for the name table.
**
**  VERSION: DCE 1.0
**
*/

#include <nidl.h>
#include <ctype.h>

#include <errors.h>
#include <nametbl.h>
#include <namtbpvt.h>
#include <nidlmsg.h>

extern NAMETABLE_id_t NAMETABLE_root;
extern NAMETABLE_temp_name_t * NAMETABLE_temp_chain;

/******************************************************************************/
/*                                                                            */
/*                  N A M E T A B L E          D U M P E R S                  */
/*                                                                            */
/******************************************************************************/

/*
 * Function: Dumps the binding information for a node of the nametable tree.
 *
 * Inputs:   Node of the tree to be dumped.
 *
 * Outputs:  None.
 *
 * Functional Value: None.
 *
 * Notes:    None.
 *
 */

static void NAMETABLE_dump_bindings_4_node
(
    NAMETABLE_binding_n_t * pp
)
{
    NAMETABLE_binding_n_t * p;

    p = pp;

    while (p != NULL) {
        printf ("\tBinding node at: %p \n", p);
        printf ("\t    bindingLevel: %d\n", p->bindingLevel);
        printf ("\t    theBinding: %p \n", p->theBinding);
        printf ("\t    nextBindingThisLevel: %p \n", p->nextBindingThisLevel);
        printf ("\t    oldBinding: %p \n", p->oldBinding);
        printf ("\t    boundBy: \"%s\" ( %p )\n\n", p->boundBy->id, p->boundBy);
        if (p->oldBinding != NULL) {
            printf ("\n");
        };
        p = p->oldBinding;
    };
}

/*
 * Function: Dumps a node of the nametable tree.
 *
 * Inputs:   Node of the tree to be dumped.
 *
 * Outputs:  Text is output to stdout.
 *
 * Functional Value: None.
 *
 * Notes:    None.
 *
 */

static void NAMETABLE_dump_node
(
    NAMETABLE_id_t node
)
{
    printf ("\n\"%s\" ( %p ) :\n",        /* "FOO" (0023ad8C) : */
            node->id,                   /* The id string */
            (char *) node );            /* The address of the node */

    if (node->parent != NULL) {
        printf ("        Parent: ( %p ) \"%s\"\n", /*     Parent:  ( 01234abc ) "bar" */
                node->parent,             /* The address of the parent */
                node->parent->id);        /* The id string of parent */
    } else {
        printf ("        *** NAMETABLE ROOT ***\n");     /* Handle the NULL case. */
    };

    if (node->left != NULL) {
        printf ("        Left:  ( %p ) \"%s\"\n", /*     Left:  ( 01234abc ) "bar" */
                node->left,             /* The address of the left child */
                node->left->id);        /* The id string of l. child */
    } else {
        printf ("        Left:  NULL\n");     /* Handle the NULL case. */
    };

    if (node->right != NULL) {
        printf ("        Right: ( %p ) \"%s\"\n", /*     Right: (01234abc) "bar" */
                node->right,            /* The address of the right child */
                node->right->id);       /* The id string of r. child */
    } else {
        printf ("        Right: NULL\n");   /* Handle the NULL case. */
    };

    if (node->bindings != NULL) {
        printf ("    Head of binding chain : %p \n",
                node->bindings);
        printf ("    Binding information for \"%s\"\n",
                node->id);
        NAMETABLE_dump_bindings_4_node (node->bindings);
    } else {
        printf ("    No binding chain\n");
    };

    if (node->tagBinding != NULL) {
        printf ("    Structure with this tag: %p \n",
                node->tagBinding);
        printf ("    Tag binding information for \"%s\"\n",
                node->id);
        NAMETABLE_dump_bindings_4_node (node->tagBinding);
    } else {
        printf ("    No structures with this tag\n");
    };
}

/*
 * Function: Recursively dumps all the nodes of a nametable tree.
 *           First dumps the left subtree bottom up, then the root node,
 *           then the right subtree, resulting in an alphabetical dump.
 *
 * Inputs:   Root node of the tree to be dumped.
 *
 * Outputs:  None.
 *
 * Functional Value: None.
 *
 * Notes:    None.
 *
 */

static void NAMETABLE_dump_nodes
(
    NAMETABLE_id_t node
)
{
    if (node->left != NULL) {
        NAMETABLE_dump_nodes (node->left);
    };

    NAMETABLE_dump_node (node);

    if (node->right != NULL) {
        NAMETABLE_dump_nodes (node->right);
    };
}

/*
 * Function: Dump the list of temporary name table nodes.
 *
 * Inputs:   NAMETABLE_temp_chain (Implicit)
 *
 * Outputs:
 *
 * Functional Value:
 *
 * Notes:
 *
 */

static void NAMETABLE_dump_temp_node_list (void)
{
NAMETABLE_temp_name_t * tb;

    if (!NAMETABLE_temp_chain) {
        printf ("\n\nThere are no temporary names.\n");
        return;
    } else {
        printf ("\n\nTemporary name chain:\n");
    }

    for (tb = NAMETABLE_temp_chain; tb; tb = tb->next) {
        printf ("    Chain block: %p NT node ( %p ): \"%s\"\n",
            tb, tb->node, tb->node->id );
    }

}

/*
 * Function: Dump a name table in human-readable form.
 *
 * Inputs:   name_table - the table to be dumped. (Implicit)
 *
 * Outputs:
 *
 * Functional Value:
 *
 * Notes:
 *
 */

void NAMETABLE_dump_tab (void)
{
    NAMETABLE_dump_nodes( NAMETABLE_root );
    NAMETABLE_dump_temp_node_list();
}
