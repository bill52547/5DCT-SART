function updatedImg = SART_cuda(img, para, iter_para)
%SART_CUDA update the image with SART, consisiting component of deformations in 5DCT model. 
%In regular SART iterative reconstruction methods, there are only projectors, backprojectors and some other basic arithemtic operators. However in this implementation of SART, we added some deformation operators, with whom a regular image in Cartesian coordinates could be deformed into another Cartesian, connected by DVFs (deformation vector fields).
%The most area for this kind of structure is motivated by irregular breathing of lung-cancer patients, who hardly breathes regular. 
%
% Syntax:  updatedImg = SART_cuda(img, para, iter_para)
%
% Inputs:
%    img - initial guess or the pre-staged image of 1st reference bin. 
%    para - parameters those are indenpendent with projection angles
%    iter_para - paratmeters those are dependent with projection angles, and the model parameters.
%
% Outputs:
%    output1 - Description
%    output2 - Description
%
% Example: 
%    Line 1 of example
%    Line 2 of example
%    Line 3 of example
%
% Other m-files required: none
% Subfunctions: none
% MAT-files required: none
%
% See also: OTHER_FUNCTION_NAME1,  OTHER_FUNCTION_NAME2

% Author: FirstName FamilyName
% Work address
% email: 
% Website: http://www.
% May 2004; Last revision: 12-May-2004

%------------- BEGIN CODE --------------

Enter your executable matlab commands here

%------------- END OF CODE --------------
%Please send suggestions for improvement of the above template header 
%to Denis Gilbert at this email address: gilbertd@dfo-mpo.gc.ca.
%Your contribution towards improving this template will be acknowledged in
%the "Changes" section of the TEMPLATE_HEADER web page on the Matlab
%Central File Exchange