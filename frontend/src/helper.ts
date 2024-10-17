// Define the type for the result object
type  SupplierInfo =  {
    supplier: string;
    product: string;
    price: string;
    location: string;
    image: string;
  }

  export const convertStringToObject = (str: string): SupplierInfo => {

    const keyValuePairs = str.split(', ');


    const result = keyValuePairs.reduce((acc, pair) => {

      const [key, value] = pair.split(': ');

      const formattedKey = key.toLowerCase().replace(/\s+/g, '_') as keyof SupplierInfo;

      acc[formattedKey] = value;

      return acc;
    }, {} as SupplierInfo);

    return result;
  };

  const inputString = "Supplier: Private Seller: Nkiriyehe V, Product: iPhone 13 Black color Storage 128GB, Price: RWF 430,000, Location: Kigali, Rwanda, Image: https://i.ebayimg.com/thumbs/images/g/CdwAAOSwr1Jlsn-y/s-l140.jpg";

  const resultObject = convertStringToObject(inputString);

  console.log(resultObject);
