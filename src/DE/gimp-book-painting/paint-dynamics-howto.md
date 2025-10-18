# How to use the GIMP Paint Dynamics
The main scope is describe the guidelines to emulate some paint techniques.

Basically the paint dynamic emulates the technique and the brush emulates the stain quality and characteristics of the medium, but this is not always true, what I am saying is: that in certain cases some stain marks could be alone and without paint dynamics represents a good mark for a dry medium, but with limitations to use effectively to paint. In general, a good paint dynamics dedicated to a paint technique, e.g., dry media, it could work with many stain marks also with marks thought or pulled by another medium as watercolor, for instance.

## Methodology
To know where until is possible to push the emulation, via paint dynamics, is necessary a method or approach to testing different variables and parameters on paint dynamics. In theory, the paint dynamics could be thought to simplify the brush set and at minimum, we can have one unique brush for all techniques with different paint dynamics, one for each technique, eg. The method or approach could be exemplified so:

_Only we need a new brush or a special when the paint dynamics are not capable of emulating effectively a specific pictorial technique._

## Dry Media
Dry Media is normally associated with drawing tools, such as *graphite pencils*, *charcoal*, *pastel* and *crayon*.
These drawing tools depending of the material which is build releases onto support surface, in general paper, creating a characteristic mark.
The dry media is probably the most ancestral way of visual communication, for instance, the Lascaux's and Altamira's caves are a great example of that.
The dry media are the main tools to drawing and sketching on the most common artistic tasks.

### Dry Media Techniques
Here I will try study how these aspects could be emulated via paint dynamics, for instance, the 'outline' technique is very simple to emulate but 'gradate' and 'layering' are just more complex, because they need blending capabilities. Another aspect is also how each paint tool could be used to emulate these dry media techniques, for instance, the first four techniques we can use practically pencil, paintbrush and airbrush.

The more common techniques are:

1. Outline
2. Hatching
3. Cross Hatching
4. Pointillism
5. Gradate (Back and Fort Stroke. Scumbling, )
6. Layering

#### Paint Tools and Dry Media
For the traditional techniques as 'Outline', 'Hatching' and 'Cross Hatching' is possible to use all main paint tools to drawing: pencil and paintbrush. These tools are interesting also to 'Pointillism', 'Gradate' and 'Layering' using the blend modes (Darken Only is my preferred), but the strokes are a bit uniform, some times it seems an opaque technique as 'Tempera' or 'Gouache'. Instead, using the 'Airbrush' or the 'Smudge', but mainly the 'Airbrush', because it has the Rate and Flow controls and the blending is more effective and well done than other tools.

### Dry Media Marks and Emulation techniques
In general all dry media are similar marks among them with variations on the granularity, e.g., the graphite pencil has different hardness and is possible to see the charcoal aspect with graphite pencils with hardness around 4B~6B.
Another important factor on the mark of dry media tools depending of kind support, for instance, a pressed paper when is compared with a textured paper (like as Ingres and Murillo papers, etc) has a grain more small and the material released of the dry tool will have an aspect different.
In Digital Painting we don't have support concept while we are paint or drawing, except our works are printed. For this reason, many of digital tools or presets to emulate drawing tools must have also the emulation of support.

#### The main aspects of Dry Media
The main aspects of dry media on real drawing/painting tasks are:
1. Tip Aspect shapes of the tool. Each media tool has its own natural shapes or the capacity to produce different kind shapes following the orientation and gesture of the hand. For instance, the pencil and crayon has a similar tip: round when used with 60~90 degrees in relation of surface support and prismatic when is heavily inclined.
2. Tilt/Pressure - mark variations along the stroke, for instance, when the tool is inclined/pressed, the stroke could modify how is the original mark is printed with the tool at 90 degrees with same weight. The quality mark can be modified mainly on the size, opacity/hardness, aspect and texture variations.
3. Texture - this aspect on dry media is in great part derivative of two components: the media material, also its components and, the support utilized (paper, canvas, wood, etc.). Supposing a perfect and sharp surface, e.g. the glass, is very difficult to create a stroke mark and to remain material of media attached it.
4. Blending capabilities - on some dry medias we have interesting blending skills, for instance, the colored pencils have this possibility, also the crayon. Is possible to use, on pastel and charcoal, the 'sfumino' to create blending effect.

The parameters that we can use to emulate these aspects on digital painting could be:
1. Aspect tip variations;
2. Spread or jittery of the mark tip;
3. Angle (size, opacity, hardness, );
4. Pressure (size, opacity, mark texture, );
5. Velocity and Acceleration (size, opacity, mark texture, );
6. Blending skill [color (fg/bg); opacity x tilt, hardness x tilt, Rate, Flow and Blend Modes)

#### Troubleshooting of knew issues

### Kind of Brushes
GIMP has two types of brushes: the parametric and raster brushes (.gbr and .gih).
The parametric brush is effectively a vector brush and it has the great possibilities and flexibility for many tasks in different tasks and branches (graphic design, photography, digital painting, etc...)
The raster brush is also versatile and has great pictorial qualities but is more complex to build because is managed as image.
Probably the choice, between raster and parametric brush, is done by the kind of task, the painter style and of his/her artistic background/knowledge/taste.

#### Parametric Brushes
Is possible to use the parametric brush for many of the traditional tasks of the painting, for instance, sketching, drafts, studies, etc. The parametric brush compared to .gih/.gbr brush is a bit poor, mainly when the artist  has a style more sophisticated, or in other words, an own style.

The parametric brushes, in general, are working very well in many aspects, for instance:
1. the processing is more fast when we comparing with .gih/.gbr brushes.
2. the hardness is applied correctly on the shapes.
3. has an own editor... instead the .gih/.gbr doesn't has.

#### Raster Brushes (.gbr and .gih)
The raster brushes are more knew for the majority of users and because mainly, the usage is more common or popular. When we compare with parametric brushes we have some advantages and not. In general, they are more interesting for the artists where their styles follow an texture finishing not usual or classic. With .gih/.gbr brushes, we can use any stain to emulate many real aspects of traditional techniques and they can be used in many ways: like as classic painter or like as Van Gogh. If you use the .gih brush, for instance, is more sophisticated than a .gbr brush because is possible to have more aspects and behaviors together and, this amplified in another way, more advanced, than the .gbr brush, that is essentially a static brush.

Some against of raster brushes are:
1. the hardness is not applied correctly/oddly on gih/gbr, see: https://gitlab.gnome.org/GNOME/gimp/issues/1081
2. has not a own editor, instead is managed like as an image and plugins to create the binary archives.

## References:
1. [Dry Media Definition](https://helenatkn.wordpress.com/2016/03/15/dry-media-wet-media-digital-media-individual-research/ "Dry Media Definition")
2. [Steven Saitzyk Website Project- Adjunt Professor at Art Center of Design in Pasadena](http://www.trueart.info/?page_id=289 "True Art Information - Information About Artist's Meterials")
3. [Pigments History](http://www.jcsparks.com/painted/pigment-chem.html "Pigments History")
4. [Pigment Catalog](http://www.webexhibits.org/pigments/intro/pigments.html "Pigments Trough the Ages")
5. [Pigments on Wikipedia](https://en.wikipedia.org/wiki/Pigment "Pigment History and References")
