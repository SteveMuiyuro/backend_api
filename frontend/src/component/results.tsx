import { IoLocationOutline } from 'react-icons/io5';
import { RiPhoneLine } from 'react-icons/ri';
import { MdOutlineEmail } from 'react-icons/md';
import { IoIosLink } from 'react-icons/io';
import { Supplier } from '../../src/types';

type ResultsProps = {
  result: Supplier;
};

function Results({ result }: ResultsProps) {


  return (
    <div className='text-gray-800 flex gap-[50px] ml-20'>
      <img
        className="object-cover w-[100px] h-[150px]"
        src={result.product_image_url}
        alt={result.product_name}
      />
      <div className="flex flex-col">
          <p>{result.product_name}</p>
          <p className="font-bold">{result.price}</p>
          <p className="font-medium py-1">{result.supplier}</p>
          <div className="grid grid-cols-2 gap-5 mb-1">
            <div className="flex justify-start items-center gap-2">
              <IoLocationOutline className='flex  flex-shrink-0' />
              <p className="text-sm">{result.location}</p>
            </div>
            <div className="flex justify-start items-center gap-2">
              <RiPhoneLine className='flex  flex-shrink-0'/>
              <p className="text-sm break-words">{result.supplier_contact} </p>
            </div>
          </div>
          <div className="grid grid-cols-2 gap-4">
            <div className="flex justify-start items-center gap-2">
              <MdOutlineEmail className="flex  flex-shrink-0"/>
              <p className="text-sm">{result.email}</p>
            </div>
            <div className="flex justify-start items-center gap-2 w-full">
              <IoIosLink className="flex flex-shrink-0"/>
              <a href={result.website} target='_blank' className="text-sm break-words" style={{ wordBreak: 'break-all' }}>
                {result.website}
              </a>
            </div>
          </div>
      </div>
  </div>
  );
}

export default Results;
