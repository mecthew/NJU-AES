————————————————> 1. License agreement

 Copyright (C) 2015  Radu Tudor Ionescu, Marius Popescu
 
 This package contains free software: you can redistribute it and/or modify it under
 the terms of the GNU General Public License as published by the Free Software
 Foundation, either version 3 of the License, or any later version.
 
 This program is distributed in the hope that it will be useful, but WITHOUT ANY
 WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
 PARTICULAR PURPOSE.  See the GNU General Public License for more details.
 
 You should have received a copy of the GNU General Public License along with this
 program (see COPYING.txt package file). If not, see <http://www.gnu.org/licenses/>.


————————————————> 2. Citation
 
 Please cite the corresponding works (see citation.bib package file to obtain the
 BibTex) if you use this software (or a modified version of it) in any scientific
 work:

 [1] Ionescu, Radu Tudor, Popescu, Marius, and Cahill, Aoife. Can characters 
     reveal your native language? a language-independent approach to native 
     language identification. Proceedings of EMNLP, pp. 1363–1373, October 2014.

 [2] Popescu, Marius and Ionescu, Radu Tudor. The Story of the Characters, 
     the DNA and the Native Language. Proceedings of the Eighth Workshop on 
     Innovative Use of NLP for Building Educational Applications, pp. 270–278,
     June 2013.

 [3] Ionescu, Radu Tudor and Popescu, Marius. Knowledge Transfer between Computer
     Vision and Text Mining: Similarity-based Learning Approaches. Springer, 2016.


————————————————> 3. Software Website:

 This software is available at: http://string-kernels.herokuapp.com/


————————————————> 4. Usage
 
 The software is written in Java. It has no dependencies on third-party libraries.
 It is straight forward to compile and run the code. A sample file is also provided
 with the package. 
 
 Source files:
 BlendedIntersectionStringKernel.java
 BlendedPresenceStringKernel.java
 BlendedSpectrumStringKernel.java
 ComputeStringKernel.java

 Java compile command: 
 javac ComputeStringKernel.java

 Running in terminal:
 1. To print the program description, run:
    $ java ComputeStringKernel

 2. For an example of usage, run one of the following commands:
    $ java ComputeStringKernel intersection 3 5 sentences.txt K_intersection_3-5.txt
    $ java ComputeStringKernel presence 3 5 sentences.txt K_presence_3-5.txt
    $ java ComputeStringKernel spectrum 3 5 sentences.txt K_spectrum_3-5.txt


————————————————> 5. Feedback and suggestions
 
 Send an e-mail to: raducu[dot]ionescu{at}gmail[dot].com
